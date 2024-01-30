import pickle
import wandb
import warnings
from collections import defaultdict

from modelling.model import build_model
from utils.checkpoint_average import average_checkpoints

warnings.filterwarnings("ignore")
import argparse
import os
import sys

sys.path.append(os.getcwd())  # slt dir
import torch
from utils.misc import (
    get_logger,
    set_seed,
    load_config,
    make_logger, move_to_device,
    neq_load_customized
)
from dataset.Dataloader import build_dataloader
from utils.progressbar import ProgressBar
from utils.metrics import bleu, rouge, wer_list
from utils.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014


def evaluation(
        cfg,
        model,
        val_dataloader,
        tb_writer=None,
        wandb_run=None,
        epoch=None,
        global_step=None,
        generate_cfg={},
        save_dir=None,
        do_translation=True,
        do_recognition=True
):
    # print()
    logger = get_logger()
    logger.info(generate_cfg)
    if os.environ.get('enable_pbar', '1') == '1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc='Validation')
    else:
        pbar = None
    if epoch is not None:
        logger.info(
            'Evaluation epoch={} validation examples #={}'.format(epoch, len(val_dataloader.dataset))
        )
    elif global_step is not None:
        logger.info(
            'Evaluation global step={} validation examples #={}'.format(global_step, len(val_dataloader.dataset))
        )
    model.eval()
    total_val_loss = defaultdict(int)

    results = defaultdict(dict)
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            # forward -- loss
            batch = move_to_device(batch, cfg['device'])
            forward_output = model.forward(is_train=False, **batch)
            for k, v in forward_output.items():
                if '_loss' in k:
                    total_val_loss[k] += v.item()
            if do_recognition:  # wer
                # rgb/keypoint/fuse/ensemble_last_logits
                for k, gls_logits in forward_output.items():
                    if 'gloss_logits' not in k or gls_logits is None:
                        continue
                    logits_name = k.replace('gloss_logits', '')
                    if logits_name in ['rgb_', 'keypoint_', 'fuse_', 'ensemble_last_', 'ensemble_early_', '']:
                        if logits_name == 'ensemble_early_':
                            input_lengths = forward_output['aux_lengths']['rgb'][-1]
                        else:
                            input_lengths = forward_output['input_lengths']
                        ctc_decode_output = model.predict_gloss_from_logits(
                            gloss_logits=gls_logits,
                            beam_size=generate_cfg['recognition']['beam_size'],
                            input_lengths=input_lengths
                        )
                        batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output)
                        for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                            results[name][f'{logits_name}gls_hyp'] = ' '.join(gls_hyp).upper() \
                                if model.gloss_tokenizer.lower_case else ' '.join(gls_hyp)
                            results[name]['gls_ref'] = gls_ref.upper() \
                                if model.gloss_tokenizer.lower_case else gls_ref
                            # print(logits_name)
                            # print(results[name][f'{logits_name}gls_hyp'])
                            # print(results[name]['gls_ref'])

                    else:
                        print(logits_name)
                        raise ValueError
                # multi-head
                if 'aux_logits' in forward_output:
                    for stream, logits_list in forward_output['aux_logits'].items():  # ['rgb', 'keypoint]
                        lengths_list = forward_output['aux_lengths'][stream]  # might be empty
                        for i, (logits, lengths) in enumerate(zip(logits_list, lengths_list)):
                            ctc_decode_output = model.predict_gloss_from_logits(
                                gloss_logits=logits,
                                beam_size=generate_cfg['recognition']['beam_size'],
                                input_lengths=lengths)
                            batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output)
                            for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                                results[name][f'{stream}_aux_{i}_gls_hyp'] = ' '.join(gls_hyp).upper() \
                                    if model.gloss_tokenizer.lower_case else ' '.join(gls_hyp)

            if do_translation:
                generate_output = model.generate_txt(
                    transformer_inputs=forward_output['transformer_inputs'],
                    generate_cfg=generate_cfg['translation']
                )
                if forward_output.get("posterior_encoder_outputs", None) is not None:
                    forward_output['transformer_inputs'].update(
                        {"encoder_outputs": forward_output['posterior_encoder_outputs']}
                    )
                    forward_output['posterior_decoded_sequences'] = model.generate_txt(
                        transformer_inputs=forward_output['transformer_inputs'],
                        generate_cfg=generate_cfg['translation']
                    )['decoded_sequences']
                # Tips: to be compatible with version without posterior_decoded_sequences.
                if forward_output.get("posterior_decoded_sequences", None) is None:
                    forward_output['posterior_decoded_sequences'] = batch['text']
                # decoded_sequences
                for name, txt_hyp, txt_ref, txt_ref_posterior_decoded in zip(
                        batch['name'],
                        generate_output['decoded_sequences'],
                        batch['text'],
                        forward_output['posterior_decoded_sequences']
                ):
                    results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref
                    results[name]['txt_hyp_posterior_decoded'] = txt_ref_posterior_decoded

            # misc
            if pbar:
                pbar(step)
        print()
    # logging and tb_writer
    for k, v in total_val_loss.items():
        logger.info('{} Average:{:.4f}'.format(k, v / len(val_dataloader)))
        if tb_writer:
            tb_writer.add_scalar('eval/' + k, v / len(val_dataloader), epoch if epoch is not None else global_step)
        if wandb_run:
            wandb.log({f'eval/{k}': v / len(val_dataloader)})
    # evaluation (Recognition:WER,  Translation:B/M)
    evaluation_results = {}
    if do_recognition:
        evaluation_results['wer'] = 200
        for hyp_name in results[name].keys():
            if 'gls_hyp' not in hyp_name:
                continue
            k = hyp_name.replace('gls_hyp', '')
            if cfg['data']['dataset_name'].lower() == 'phoenix-2014t':
                gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
                gls_hyp = [clean_phoenix_2014_trans(results[n][hyp_name]) for n in results]
            elif cfg['data']['dataset_name'].lower() == 'phoenix-2014':

                gls_ref = [clean_phoenix_2014(results[n]['gls_ref']) for n in results]
                gls_hyp = [clean_phoenix_2014(results[n][hyp_name]) for n in results]
            elif cfg['data']['dataset_name'].lower() in ['csl-daily', 'cslr']:
                gls_ref = [results[n]['gls_ref'] for n in results]
                gls_hyp = [results[n][hyp_name] for n in results]
            wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
            evaluation_results[k + 'wer_list'] = wer_results
            logger.info('{}WER: {:.2f}'.format(k, wer_results['wer']))
            if tb_writer:
                tb_writer.add_scalar(f'eval/{k}WER', wer_results['wer'], epoch if epoch != None else global_step)
            if wandb_run is not None:
                wandb.log({f'eval/{k}WER': wer_results['wer']})
            evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])
    if do_translation:
        txt_ref = [results[n]['txt_ref'] for n in results]
        txt_hyp = [results[n]['txt_hyp'] for n in results]
        bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=cfg['data']['level'])
        rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=cfg['data']['level'])
        logger.info(", ".join('{}: {:.2f}'.format(k, v) for k, v in bleu_dict.items()))

        txt_posterior_iter_hyp = [results[n]['txt_hyp_posterior_decoded'] for n in results]
        bleu_dict_posterior_decoded = bleu(references=txt_ref, hypotheses=txt_posterior_iter_hyp, level=cfg['data']['level'])
        rouge_score_posterior_decoded = rouge(references=txt_ref, hypotheses=txt_posterior_iter_hyp, level=cfg['data']['level'])
        logger.info(", ".join('{}: {:.2f}'.format(k, v) for k, v in bleu_dict_posterior_decoded.items())+"-Posterior")

        logger.info('ROUGE: {:.2f}'.format(rouge_score))
        logger.info('ROUGE: {:.2f}-Posterior'.format(rouge_score_posterior_decoded))
        evaluation_results['rouge'], evaluation_results['bleu'] = rouge_score, bleu_dict
        if tb_writer:
            tag = epoch if epoch is not None else global_step
            tb_writer.add_scalar('eval/BLEU4', bleu_dict['bleu4'], tag)
            tb_writer.add_scalar('eval/ROUGE', rouge_score, tag)
            tb_writer.add_scalar('eval/BLEU4_posterior_iter', bleu_dict_posterior_decoded['bleu4'], tag)
            tb_writer.add_scalar('eval/ROUGE_posterior_iter', rouge_score_posterior_decoded, tag)
        if wandb_run is not None:
            wandb.log({'eval/BLEU4': bleu_dict['bleu4']})
            wandb.log({'eval/ROUGE': rouge_score})
    # save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(save_dir, 'evaluation_results.pkl'), 'wb') as f:
            pickle.dump(evaluation_results, f)

        parent_dir = os.path.dirname(save_dir)
        if save_dir.endswith("test"):
            r = "R_{:.2f}_".format(evaluation_results['rouge'])
            b = "_".join('{}_{:.2f}'.format(k, v) for k, v in evaluation_results['bleu'].items())
            file_name = "Test_" + r + b
            with open(os.path.join(parent_dir, file_name), 'w') as f:
                f.write("")
            # posterior decode res
            r = "R_{:.2f}_".format(rouge_score_posterior_decoded)
            b = "_".join('{}_{:.2f}'.format(k, v) for k, v in bleu_dict_posterior_decoded.items())
            file_name = "Posterior_Test_" + r + b
            with open(os.path.join(parent_dir, file_name), 'w') as f:
                f.write("")
        elif save_dir.endswith("dev"):
            r = "R_{:.2f}_".format(evaluation_results['rouge'])
            b = "_".join('{}_{:.2f}'.format(k, v) for k, v in evaluation_results['bleu'].items())
            file_name = "Dev_" + r + b
            with open(os.path.join(parent_dir, file_name), 'w') as f:
                f.write("")
            r = "R_{:.2f}_".format(rouge_score_posterior_decoded)
            b = "_".join('{}_{:.2f}'.format(k, v) for k, v in bleu_dict_posterior_decoded.items())
            file_name = "Posterior_Dev_" + r + b
            with open(os.path.join(parent_dir, file_name), 'w') as f:
                f.write("")
        else:
            # save_dir.endswith("validation"):
            pass
    return evaluation_results


def add_parser():



    return args, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CV-SLT")
    parser.add_argument("--config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    parser.add_argument("--save-subdir", default='prediction', type=str)
    parser.add_argument('--ckpt-name', default='best.ckpt', type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # =============== for scripts params ===============

    model_dir = cfg['training']['model_dir']
    set_seed(seed=cfg["training"].get("random_seed", 42))
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction.log')
    cfg['device'] = torch.device('cuda')
    model = build_model(cfg)
    # load model
    load_model_path = os.path.join(model_dir, 'ckpts', args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location='cuda')
        logger.info('Load model ckpt from ' + load_model_path)
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch = state_dict.get('epoch', 0)
        global_step = state_dict.get('global_step', 0)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    do_translation, do_recognition = cfg['task'] != 'S2G', cfg['task'] != 'G2T'
    do_recognition = (
        cfg['task'] not in ['G2T', 'S2T_glsfree'] and cfg['model']['recognition_weight'] > 0.
        if not cfg.get("do_recognition", False) else True
    )
    for split in ['dev', 'test']:
        logger.info('Evaluate on {} set'.format(split))
        dataloader, sampler = build_dataloader(
            cfg, split,
            model.text_tokenizer,
            model.gloss_tokenizer
        )
        evaluation_result = evaluation(
            cfg=cfg,
            model=model,
            val_dataloader=dataloader,
            epoch=epoch,
            global_step=global_step,
            generate_cfg=cfg['testing']['cfg'],
            save_dir=os.path.join(model_dir, args.save_subdir, split),
            do_translation=do_translation,
            do_recognition=do_recognition,
        )


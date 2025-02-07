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
import tqdm

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
            # in case of OOM during test, place the amp code here.
            # if cfg['training']['amp']:
            #     with torch.cuda.amp.autocast():
            #         forward_output = model.forward(is_train=False, **batch)
            # else:
            #     forward_output = model.forward(is_train=False, **batch)
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
                    forward_output['first_decoded_sequences'] = model.generate_txt(
                        transformer_inputs=forward_output['transformer_inputs'],
                        generate_cfg=generate_cfg['translation']
                    )['decoded_sequences']
                # Tips: to be compatible with version without first_decoded_sequences.
                if forward_output.get("first_decoded_sequences", None) is None:
                    forward_output['first_decoded_sequences'] = batch['text']
                # decoded_sequences
                for name, txt_hyp, txt_ref, txt_ref_first_decoded in zip(
                        batch['name'],
                        generate_output['decoded_sequences'],
                        batch['text'],
                        forward_output['first_decoded_sequences']
                ):
                    results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref
                    results[name]['txt_hyp_first_decoded'] = txt_ref_first_decoded

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

        txt_first_iter_hyp = [results[n]['txt_hyp_first_decoded'] for n in results]
        bleu_dict_first_decoded = bleu(references=txt_ref, hypotheses=txt_first_iter_hyp, level=cfg['data']['level'])
        rouge_score_first_decoded = rouge(references=txt_ref, hypotheses=txt_first_iter_hyp, level=cfg['data']['level'])
        logger.info(", ".join('{}: {:.2f}'.format(k, v) for k, v in bleu_dict_first_decoded.items())+"-Posterior")

        logger.info('ROUGE: {:.2f}'.format(rouge_score))
        logger.info('ROUGE: {:.2f}-Posterior'.format(rouge_score_first_decoded))
        evaluation_results['rouge'], evaluation_results['bleu'] = rouge_score, bleu_dict
        if tb_writer:
            tag = epoch if epoch is not None else global_step
            tb_writer.add_scalar('eval/BLEU4', bleu_dict['bleu4'], tag)
            tb_writer.add_scalar('eval/ROUGE', rouge_score, tag)
            tb_writer.add_scalar('eval/BLEU4_first_iter', bleu_dict_first_decoded['bleu4'], tag)
            tb_writer.add_scalar('eval/ROUGE_first_iter', rouge_score_first_decoded, tag)
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
            # first decode res
            r = "R_{:.2f}_".format(rouge_score_first_decoded)
            b = "_".join('{}_{:.2f}'.format(k, v) for k, v in bleu_dict_first_decoded.items())
            file_name = "Posterior_Test_" + r + b
            with open(os.path.join(parent_dir, file_name), 'w') as f:
                f.write("")
        elif save_dir.endswith("dev"):
            r = "R_{:.2f}_".format(evaluation_results['rouge'])
            b = "_".join('{}_{:.2f}'.format(k, v) for k, v in evaluation_results['bleu'].items())
            file_name = "Dev_" + r + b
            with open(os.path.join(parent_dir, file_name), 'w') as f:
                f.write("")
            r = "R_{:.2f}_".format(rouge_score_first_decoded)
            b = "_".join('{}_{:.2f}'.format(k, v) for k, v in bleu_dict_first_decoded.items())
            file_name = "Posterior_Dev_" + r + b
            with open(os.path.join(parent_dir, file_name), 'w') as f:
                f.write("")
        else:
            # save_dir.endswith("validation"):
            pass
    return evaluation_results


def add_parser():
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument("--config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    parser.add_argument("--avg", action="store_true", help="average ckpt")
    parser.add_argument("--model-dir", default='as_config', type=str)
    parser.add_argument("--save-subdir", default='prediction', type=str)
    parser.add_argument('--ckpt-name', default='best.ckpt', type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action='store_true', default=False)

    # model hyper params
    parser.add_argument("--do-recognition", action='store_true', help="whether do recognition on valid set.")
    parser.add_argument("--freeze", action='store_true', help="whether freeze visual head of recognition net.")
    parser.add_argument("--mode", type=str, default="original")
    parser.add_argument("--recognition-weight", type=float, default=1.0)
    parser.add_argument("--kl-factor", type=float, default=1.0)  # act as kl factor simultaneously
    parser.add_argument("--temperature", type=float, default=1.0)  # act as kl factor simultaneously
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--combine-type", type=str, default="residual",
                        choices=['gate_residual', 'residual', 'mixup', 'concatenate'])

    parser.add_argument("--variator-layers", type=int, default=1)
    parser.add_argument("--variator-type", type=str, default="cross_attention")
    parser.add_argument("--norm", type=str, default="prefix", choices=["prefix", "middle", "postfix", "none"])
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--mlp", action='store_true', default=False)
    # parser.add_argument("--variator-type", type=str)
    ##
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_dir != "as_config":
        cfg['training']['model_dir'] = args.model_dir
    cfg['training']['batch_size'] = args.batch_size
    cfg['training']['random_seed'] = args.seed
    cfg['training']['num_workers'] = args.num_workers
    cfg['training']['amp'] = args.amp
    cfg['model']['mode'] = args.mode
    cfg['do_recognition'] = args.do_recognition
    cfg['model']['recognition_weight'] = args.recognition_weight * float(args.do_recognition)
    if cfg['do_recognition']:
        cfg['model']['RecognitionNetwork']['freeze'] = args.freeze
        cfg['model']['RecognitionNetwork']['visual_head']['freeze'] = args.freeze
    if cfg['model']['mode'] in ["variational", "autoencoder"]:
        cfg['model']['VariationalNetwork']['kl_factor'] = args.kl_factor
        cfg['model']['VariationalNetwork']['temperature'] = args.temperature
        cfg['model']['VariationalNetwork']['latent_dim'] = args.latent_dim
        cfg['model']['VariationalNetwork']['combine_type'] = args.combine_type
        cfg['model']['VariationalNetwork']['variator_layers'] = args.variator_layers
        cfg['model']['VariationalNetwork']['norm'] = args.norm
        cfg['model']['VariationalNetwork']['gamma'] = args.gamma
        cfg['model']['VariationalNetwork']['variator_type'] = args.variator_type
        cfg['model']['VariationalNetwork']['mlp'] = args.mlp
    return args, cfg


if __name__ == "__main__":

    args, cfg = add_parser()
    model_dir = args.model_dir

    set_seed(seed=cfg["training"].get("random_seed", 42))
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='tsne.log')
    cfg['device'] = torch.device('cuda')
    model = build_model(cfg)
    # load model
    load_model_path = os.path.join(model_dir, 'ckpts', args.ckpt_name)
    if os.path.isfile(load_model_path):
        # TODO(rzhao): check out here for avg ckpt.
        if args.avg:
            ckpts_dir = os.path.join(model_dir, 'ckpts')
            ckpt_names = os.listdir(ckpts_dir)
            ckpt_names.remove(args.ckpt_name)
            ckpt_path_list = [os.path.join(ckpts_dir, ckpt_name) for ckpt_name in ckpt_names]
            state_dict = average_checkpoints(ckpt_path_list)
            logger.info('Load model ckpts from {}, {} ckpts in total.'.format(ckpts_dir, len(ckpt_path_list)))
        else:
            state_dict = torch.load(load_model_path, map_location='cuda')
            logger.info('Load model ckpt from ' + load_model_path)
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch = state_dict.get('epoch', 0)
        global_step = state_dict.get('global_step', 0)

    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0

    dataloader, sampler = build_dataloader(
        cfg, 'dev',
        model.text_tokenizer,
        model.gloss_tokenizer
    )
    model.eval()
    saved_dict = {}
    sign_features = torch.tensor([])
    mapped_features = torch.tensor([])
    text_embeddings = torch.tensor([])
    encoder_hidden_states = [torch.tensor([]) for _ in range(13)]
    prior_encoder_hidden_states = [torch.tensor([]) for _ in range(13)]
    posterior_encoder_hidden_states_sign = [torch.tensor([]) for _ in range(13)]
    posterior_encoder_hidden_states_text = [torch.tensor([]) for _ in range(13)]
    pbar = ProgressBar(n_total=len(dataloader), desc='Validation')
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            pbar(step)
            batch = move_to_device(batch, cfg['device'])
            recognition_inputs = batch['recognition_inputs']
            recognition_outputs = model.recognition_network.forward(is_train=False, **recognition_inputs)
            sign_feature = recognition_outputs['gloss_feature']
            mapped_feature = model.vl_mapper.forward(visual_outputs=recognition_outputs)
            translation_inputs = {
                **batch['translation_inputs'],
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths']
            }
            translation_outputs = model.translation_network.forward(
                **{
                    **batch['translation_inputs'],
                    'input_feature': mapped_feature,
                    'input_lengths': recognition_outputs['input_lengths']
                }
            )
            text = batch['translation_inputs']['labels']
            text_mask = text.ne(model.translation_network.text_tokenizer.pad_index)
            sign_mask = batch['recognition_inputs']['sgn_mask']
            text_embedding = model.translation_network.model.model.shared(text)
            mask_pad = False
            token_level = False
            if mask_pad:
                # sign feature mask
                sign_feature[~sign_mask] = 0.
                sign_feature = sign_feature.sum(dim=1) / sign_mask.sum(-1, keepdim=True)
                sign_features = torch.cat([sign_features, sign_feature.cpu()], dim=0)
                # mapped feature, i.e., wo  scale for encoder input.
                mapped_feature[~sign_mask] = 0.
                mapped_feature = mapped_feature.sum(dim=1) / sign_mask.sum(-1, keepdim=True)
                mapped_features = torch.cat([mapped_features, mapped_feature.cpu()], dim=0)

                # text embedding mask.
                text_embedding[~text_mask] = 0.
                text_embedding = text_embedding.sum(dim=1) / text_mask.sum(-1, keepdim=True)
                text_embeddings = torch.cat([text_embeddings, text_embedding.cpu()], dim=0)
            elif token_level:
                pass
            else:
                if args.mode == "variational":
                    for i, hidden_state in enumerate(translation_outputs['prior_encoder_hidden_states']):
                        prior_encoder_hidden_states[i] = torch.cat(
                            [prior_encoder_hidden_states[i], hidden_state.cpu().mean(dim=1)], dim=0
                        )
                    length = sign_mask.size(1)
                    for i, hidden_state in enumerate(translation_outputs['posterior_encoder_hidden_states']):
                        posterior_encoder_hidden_states_sign[i] = torch.cat(
                            [posterior_encoder_hidden_states_sign[i], hidden_state[:, :length].cpu().mean(dim=1)], dim=0
                        )
                        posterior_encoder_hidden_states_text[i] = torch.cat(
                            [posterior_encoder_hidden_states_text[i], hidden_state[:, length:].cpu().mean(dim=1)], dim=0
                        )
                for i, hidden_state in enumerate(translation_outputs['encoder_hidden_states']):
                    encoder_hidden_states[i] = torch.cat(
                        [encoder_hidden_states[i], hidden_state.cpu().mean(dim=1)], dim=0
                    )
                sign_features = torch.cat([sign_features, sign_feature.cpu().mean(dim=1)], dim=0)
                mapped_features = torch.cat([mapped_features, mapped_feature.cpu().mean(dim=1)], dim=0)
                text_embeddings = torch.cat([text_embeddings, text_embedding.cpu().mean(dim=1)], dim=0)
        saved_dict = {
            # "sign_mask": sign_mask,
            # "text_mask": text_mask,
            "sign_features": sign_features,
            "text_embeddings": text_embeddings,
            "mapped_features": mapped_features,
            "encoder_hidden_states": encoder_hidden_states,
            "prior_encoder_hidden_states": prior_encoder_hidden_states,
            "posterior_encoder_hidden_states_sign": posterior_encoder_hidden_states_sign,
            "posterior_encoder_hidden_states_text": posterior_encoder_hidden_states_text,
        }
        with open("/home/rzhao/tmp/tsn_s2t_dev.pkl", "wb") as f:
            pickle.dump(saved_dict, f)





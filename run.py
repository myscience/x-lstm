# import yaml
# from lightning import Trainer

from lightning.pytorch.cli import LightningCLI
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger 

# from transformers import AutoTokenizer

from xlstm import xLSTM
from xlstm.stories import TinyStoriesLightning

def cli_main():
    '''
    Main function for the training script.
    '''
    cli = LightningCLI(
        xLSTM,
        TinyStoriesLightning,
    )
    
    # with open(args.config, 'r') as f:
    #     config = yaml.safe_load(f)
        
    # tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    # tokenizer.add_special_tokens(config['tokenizer']['special_tokens'])
    
    # Load the Mamba model
    # model = xLSTM.from_config(args.config, key='mamba')
    
    # Load the dataset
    # dataset = TinyStoriesLightning.from_config(
    #     args.config,
    #     tokenizer,
    #     key='dataset'
    # )
    
    # Instantiate a logger and a callback for model checkpointing
    # callbk = ModelCheckpoint(**config['ckpt'])
    # logger = TensorBoardLogger(**config['logger'])
    
    # Load the trainer
    # trainer_conf = {
    #     **config['train'],
    #     **vars(args),
    #     'logger' : logger,
    #     'callbacks' : callbk,
    # }
    # trainer_conf.pop('config')
    # trainer = Trainer(**trainer_conf)
    
    # Train the model
    # trainer.fit(model, dataset, ckpt_path=config['misc']['resume'])

if __name__ == '__main__':
    # parser = ArgumentParser(
    #     prog='Mamba LLM Training Script',
    #     description='Train a Mamba LLM model on the Tiny Stories dataset.',
    # )
    
    # parser.add_argument(
    #     '--config',
    #     type=str,
    #     help='Path to the training configuration file.',
    # )
    
    # parser = Trainer.add_argparse_args(parser) # type: ignore
    
    # args = parser.parse_args()
    
    cli_main()
from ElectraDataModule import *
from ElectraBinaryClassification import *

if __name__ == "__main__" :
    model = ElectraClassification(learning_rate=0.0001)

    dm = ElectraClassificationDataModule(batch_size=8, train_path='english_review.pk', valid_path='val.pk',
                                    max_length=256, sep='\t', doc_col='review', label_col='label', num_workers=1)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                    dirpath='./sample_electra_binary_nsmc_chpt',
                                                    filename='ELECTRA/{epoch:02d}-{val_accuracy:.3f}',
                                                    verbose=True,
                                                    save_last=True,
                                                    mode='max',
                                                    save_top_k=-1,
                                                    )
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join('./sample_electra_binary_nsmc_chpt', 'tb_logs'))

    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        default_root_dir='./sample_electra_binary_nsmc_chpt/checkpoints',
        logger = tb_logger,
        callbacks = [checkpoint_callback, lr_logger],
        max_epochs=3,
        gpus=1
    )

    trainer.fit(model, dm)
    torch.save(model.state_dict(), "./eng_model.pth")
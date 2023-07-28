# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=[20, 25], min_lr=1e-6, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(by_epoch=True, interval=15)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, save_best='mIoU')
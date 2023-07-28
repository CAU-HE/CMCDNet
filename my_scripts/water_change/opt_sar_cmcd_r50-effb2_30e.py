_base_ = [
    './base/opt_sar_inc_vv_dataset.py',
    './base/scheduler_30e.py',
    './base/default_runtime.py'
]

optimizer = dict(type='Adam', lr=2e-4)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='CMCD',
        enc_opt_dims=[64, 256, 512, 1024, 2048],
        backbone_opt_cfg=dict(
            type='TIMMBackbone',
            model_name='resnet50',
            in_channels=4,
            out_indices=(0, 1, 2, 3, 4),
            output_stride=32,
            pretrained=True),
        enc_sar_dims=[16, 24, 48, 120, 352],
        backbone_sar_cfg=dict(
            type='TIMMBackbone',
            model_name='efficientnet_b2',
            in_channels=1,
            out_indices=(0, 1, 2, 3, 4),
            pretrained=True),
        center_block="dblock",
        side_dim=64,
        norm_cfg=norm_cfg
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=32,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0, class_weight=(1.0, 2.0)),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=(1.0, 2.0))
        ]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

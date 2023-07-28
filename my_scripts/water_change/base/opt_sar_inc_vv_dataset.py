# dataset settings
dataset_type = 'WCDataset'
data_root = '../data'
classes = ("bg", "increase")
palette=((0,0,0), (255, 0, 0))

img_scale = (256, 256)

train_pipeline = [
    dict(type='wc_LoadImageFromFile'),
    dict(type='wc_StackByChannel', keys=('img', 'aux')),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='wc_Normalize'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='wc_LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='wc_StackByChannel', keys=('img', 'aux')),
            dict(type='wc_Normalize'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=96,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/opt',
        aux_dir='train/vv',
        ann_dir='train/flood_vv',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/opt',
        aux_dir='test/vv',
        ann_dir='test/flood_vv',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/opt',
        aux_dir='test/vv',
        ann_dir='test/flood_vv',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette
    ))
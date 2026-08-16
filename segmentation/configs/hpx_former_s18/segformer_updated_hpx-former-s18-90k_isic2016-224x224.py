_base_ = [
    "../_base_/datasets/isic2016.py",
    "../_base_/schedules/schedule_90k.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(imports=["mmseg.models.backbones.timm"], allow_failed_imports=False)

norm_cfg = dict(type='SyncBN', requires_grad=True)

crop_size = (256, 256)

data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # _delete_=True,
        type="TimmModel",
        model_name="hpx_former_s18",
        drop_path_rate=0.4,
        out_indices=[0, 1, 2, 3],
        features_only=True,
        pretrained=False,
        strict=False,
    ),


    decode_head=dict(
    type='SegformerHeadUpdated',
    # decoder_channels=[32, 32, 64, 128],  # Channels for [c1, c2, c3, c4] processing
    # mixer_kernel=(7, 7),
    in_channels=[32, 32, 64, 128],  # Example: Segformer-B4 channels
    in_index=[0, 1, 2, 3],
    channels=64,
    dropout_ratio=0.1,
    num_classes=2,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    # auxiliary_head=dict(in_channels=320, num_classes=2),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=64,
        # in_channels=1024,
        in_index=2,
        channels=64,
        # channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    
    pretrained=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    constructor="LearningRateDecayOptimizerConstructor",
    paramwise_cfg={"decay_rate": 0.7, "decay_type": "layer_wise", "num_layers": 6},
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    loss_scale="dynamic",
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", power=1.0, begin=1500, end=160000, eta_min=0.0, by_epoch=False),
]

# By default, models are trained on 4 GPUs with 4 images per GPU
train_dataloader = dict(batch_size=4, num_workers=8)

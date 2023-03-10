?	oF?WI֕@oF?WI֕@!oF?WI֕@	???zP/S@???zP/S@!???zP/S@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6oF?WI֕@???'?#@1K??-lp@A?Q?y9l??I??????I@Y?qQ-???@*	??7ͯ0A2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?D?k;?@!5?7??X@)?D?k;?@15?7??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??%N?@!r????X@)???s%@1Y=?cp3??:Preprocessing2F
Iterator::Modelc??*S?@!      Y@),???t?1k:7?k`=?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 76.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???zP/S@I?.`?{?@Qp??2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???'?#@???'?#@!???'?#@      ??!       "	K??-lp@K??-lp@!K??-lp@*      ??!       2	?Q?y9l???Q?y9l??!?Q?y9l??:	??????I@??????I@!??????I@B      ??!       J	?qQ-???@?qQ-???@!?qQ-???@R      ??!       Z	?qQ-???@?qQ-???@!?qQ-???@b      ??!       JGPUY???zP/S@b q?.`?{?@yp??2@?"-
IteratorGetNext/_1_Send2Ο ?1??!2Ο ?1??"&
ConcatV2ConcatV2?6a8??!#L?@????"r
Fgradient_tape/inception_resnet_v2/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???YS???!?V~q??0"
MulMul(l??gt}?!?3????"A
#inception_resnet_v2/conv2d_2/Conv2DConv2D????P|?! I?RU??0">
RandomStandardNormalRandomStandardNormal?;??+x?!????״?"v
Lgradient_tape/inception_resnet_v2/batch_normalization_2/FusedBatchNormGradV3FusedBatchNormGradV3?????u?!?g??5??"t
Jgradient_tape/inception_resnet_v2/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3K????t?!??U?	???"v
Lgradient_tape/inception_resnet_v2/batch_normalization_1/FusedBatchNormGradV3FusedBatchNormGradV3????s?!????
???"C
#inception_resnet_v2/conv2d_4/Conv2DConv2Dx?}??s?!Z?.?????08Q      Y@Y?P???4@a8??F??S@qg@e?6@yI"l??@"?

host?Your program is HIGHLY input-bound because 76.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 
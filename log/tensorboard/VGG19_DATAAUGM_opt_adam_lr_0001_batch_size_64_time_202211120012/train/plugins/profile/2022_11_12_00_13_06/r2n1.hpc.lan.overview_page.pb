�	3�`=^�@3�`=^�@!3�`=^�@	���Y�V@���Y�V@!���Y�V@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails63�`=^�@� �$@1������i@A<��X�d�?IM�Nϻ1@YD�R���@*	n�`��9A2g
0Iterator::Model::MaxIntraOpParallelism::Prefetcheo)�3�@!�'`��X@)eo)�3�@1�'`��X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismW��3�@!t�8���X@)��B˺�?1���!FN?:Preprocessing2F
Iterator::Model��wӽ3�@!      Y@)�m�2{?1�[3��9?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 88.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9���Y�V@I��&M�A�?Q���%@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	� �$@� �$@!� �$@      ��!       "	������i@������i@!������i@*      ��!       2	<��X�d�?<��X�d�?!<��X�d�?:	M�Nϻ1@M�Nϻ1@!M�Nϻ1@B      ��!       J	D�R���@D�R���@!D�R���@R      ��!       Z	D�R���@D�R���@!D�R���@b      ��!       JGPUY���Y�V@b q��&M�A�?y���%@�"j
<gradient_tape/vgg19/block1_conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����aˤ?!����aˤ?08"-
IteratorGetNext/_1_SendW�lE�L�?!���?"&
ConcatV2ConcatV2gh	��4�?! ��dl��?"9
vgg19/block1_conv2/Conv2DConv2D�T ���?!�fYtY}�?08"h
;gradient_tape/vgg19/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput�xP���?!�u��t��?08"j
<gradient_tape/vgg19/block4_conv4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5@���?!�ݟ>͟�?08"j
<gradient_tape/vgg19/block2_conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterkg�43�?!ª��3F�?08"j
<gradient_tape/vgg19/block4_conv3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterm1@Rt�?!��I���?08"b
0Adam/Adam/update_32/ResourceApplyAdamWithAmsgradResourceApplyAdamWithAmsgrad��l&��?!�!Ww:%�?"9
vgg19/block2_conv2/Conv2DConv2D2[7]-��?!d�*M-e�?08Q      Y@Y)�"�$�I@a�v��qH@q.����@yќ�}Ǌ@"�	
host�Your program is HIGHLY input-bound because 88.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 
ńÖ
Ŗż
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8Ó

rnn_densef/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ą*"
shared_namernn_densef/kernel
x
%rnn_densef/kernel/Read/ReadVariableOpReadVariableOprnn_densef/kernel*
_output_shapes
:	Ą*
dtype0
v
rnn_densef/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namernn_densef/bias
o
#rnn_densef/bias/Read/ReadVariableOpReadVariableOprnn_densef/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_namelstm_lstm/lstm_cell/kernel

.lstm_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_lstm/lstm_cell/kernel*
_output_shapes

:@*
dtype0
¤
$lstm_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$lstm_lstm/lstm_cell/recurrent_kernel

8lstm_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_lstm/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0

lstm_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelstm_lstm/lstm_cell/bias

,lstm_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_lstm/lstm_cell/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/rnn_densef/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ą*)
shared_nameAdam/rnn_densef/kernel/m

,Adam/rnn_densef/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/kernel/m*
_output_shapes
:	Ą*
dtype0

Adam/rnn_densef/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/rnn_densef/bias/m
}
*Adam/rnn_densef/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/bias/m*
_output_shapes
:*
dtype0

!Adam/lstm_lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/lstm_lstm/lstm_cell/kernel/m

5Adam/lstm_lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/lstm_lstm/lstm_cell/kernel/m*
_output_shapes

:@*
dtype0
²
+Adam/lstm_lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/lstm_lstm/lstm_cell/recurrent_kernel/m
«
?Adam/lstm_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/lstm_lstm/lstm_cell/recurrent_kernel/m*
_output_shapes

:@*
dtype0

Adam/lstm_lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/lstm_lstm/lstm_cell/bias/m

3Adam/lstm_lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_lstm/lstm_cell/bias/m*
_output_shapes
:@*
dtype0

Adam/rnn_densef/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ą*)
shared_nameAdam/rnn_densef/kernel/v

,Adam/rnn_densef/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/kernel/v*
_output_shapes
:	Ą*
dtype0

Adam/rnn_densef/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/rnn_densef/bias/v
}
*Adam/rnn_densef/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/bias/v*
_output_shapes
:*
dtype0

!Adam/lstm_lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/lstm_lstm/lstm_cell/kernel/v

5Adam/lstm_lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/lstm_lstm/lstm_cell/kernel/v*
_output_shapes

:@*
dtype0
²
+Adam/lstm_lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/lstm_lstm/lstm_cell/recurrent_kernel/v
«
?Adam/lstm_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/lstm_lstm/lstm_cell/recurrent_kernel/v*
_output_shapes

:@*
dtype0

Adam/lstm_lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/lstm_lstm/lstm_cell/bias/v

3Adam/lstm_lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_lstm/lstm_cell/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ų$
valueĪ$BĖ$ BÄ$
ć
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
loss
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

iter

beta_1

beta_2
	decay
 learning_ratemMmN!mO"mP#mQvRvS!vT"vU#vV
 
#
!0
"1
#2
3
4
#
!0
"1
#2
3
4
 
­
$layer_regularization_losses

%layers
	variables
&layer_metrics
'non_trainable_variables
trainable_variables
(metrics
	regularization_losses
 
~

!kernel
"recurrent_kernel
#bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
 

!0
"1
#2

!0
"1
#2
 
¹
-layer_regularization_losses

.layers

/states
0layer_metrics
	variables
1non_trainable_variables
trainable_variables
2metrics
regularization_losses
 
 
 
­
3layer_regularization_losses

4layers
5layer_metrics
	variables
6non_trainable_variables
trainable_variables
7metrics
regularization_losses
][
VARIABLE_VALUErnn_densef/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErnn_densef/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
8layer_regularization_losses

9layers
:layer_metrics
	variables
;non_trainable_variables
trainable_variables
<metrics
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_lstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$lstm_lstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_lstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
 
 

=0
>1

!0
"1
#2

!0
"1
#2
 
­
?layer_regularization_losses

@layers
Alayer_metrics
)	variables
Bnon_trainable_variables
*trainable_variables
Cmetrics
+regularization_losses
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Dtotal
	Ecount
F	variables
G	keras_api
D
	Htotal
	Icount
J
_fn_kwargs
K	variables
L	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

F	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

K	variables
~
VARIABLE_VALUEAdam/rnn_densef/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/rnn_densef/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_lstm/lstm_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/lstm_lstm/lstm_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_lstm/lstm_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/rnn_densef/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/rnn_densef/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_lstm/lstm_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/lstm_lstm/lstm_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_lstm/lstm_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:’’’’’’’’’*
dtype0* 
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm_lstm/lstm_cell/kernellstm_lstm/lstm_cell/bias$lstm_lstm/lstm_cell/recurrent_kernelrnn_densef/kernelrnn_densef/bias*
Tin

2*
Tout
2*'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_26534
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%rnn_densef/kernel/Read/ReadVariableOp#rnn_densef/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp.lstm_lstm/lstm_cell/kernel/Read/ReadVariableOp8lstm_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp,lstm_lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/rnn_densef/kernel/m/Read/ReadVariableOp*Adam/rnn_densef/bias/m/Read/ReadVariableOp5Adam/lstm_lstm/lstm_cell/kernel/m/Read/ReadVariableOp?Adam/lstm_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp3Adam/lstm_lstm/lstm_cell/bias/m/Read/ReadVariableOp,Adam/rnn_densef/kernel/v/Read/ReadVariableOp*Adam/rnn_densef/bias/v/Read/ReadVariableOp5Adam/lstm_lstm/lstm_cell/kernel/v/Read/ReadVariableOp?Adam/lstm_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp3Adam/lstm_lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_28478
¹
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamernn_densef/kernelrnn_densef/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_lstm/lstm_cell/kernel$lstm_lstm/lstm_cell/recurrent_kernellstm_lstm/lstm_cell/biastotalcounttotal_1count_1Adam/rnn_densef/kernel/mAdam/rnn_densef/bias/m!Adam/lstm_lstm/lstm_cell/kernel/m+Adam/lstm_lstm/lstm_cell/recurrent_kernel/mAdam/lstm_lstm/lstm_cell/bias/mAdam/rnn_densef/kernel/vAdam/rnn_densef/bias/v!Adam/lstm_lstm/lstm_cell/kernel/v+Adam/lstm_lstm/lstm_cell/recurrent_kernel/vAdam/lstm_lstm/lstm_cell/bias/v*$
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_28562¦
łX

D__inference_lstm_cell_layer_call_and_return_conditional_losses_28233

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOp§
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1t
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:’’’’’’’’’2
mul|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2t
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_1]
add_3AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3t
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_2Ō
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addā
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»
Ŗ
'__inference_model_1_layer_call_fn_26493
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallū
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_264802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ā
Ę
lstm_lstm_while_cond_26890 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
less_lstm_lstm_strided_slice_17
3lstm_lstm_while_cond_26890___redundant_placeholder07
3lstm_lstm_while_cond_26890___redundant_placeholder17
3lstm_lstm_while_cond_26890___redundant_placeholder27
3lstm_lstm_while_cond_26890___redundant_placeholder3
identity
b
LessLessplaceholderless_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ż
E
)__inference_flatten_1_layer_call_fn_28111

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_263112
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
’
ė
while_body_25544
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_25568_0
lstm_cell_25570_0
lstm_cell_25572_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_25568
lstm_cell_25570
lstm_cell_25572¢!lstm_cell/StatefulPartitionedCall·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemü
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_25568_0lstm_cell_25570_0lstm_cell_25572_0*
Tin

2*
Tout
2*M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_251322#
!lstm_cell/StatefulPartitionedCallÖ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3¦

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4¦

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"$
lstm_cell_25568lstm_cell_25568_0"$
lstm_cell_25570lstm_cell_25570_0"$
lstm_cell_25572lstm_cell_25572_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
łX

D__inference_lstm_cell_layer_call_and_return_conditional_losses_28319

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOp§
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1t
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:’’’’’’’’’2
mul|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2t
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_1]
add_3AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3t
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_2Ō
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addā
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ķ

__inference_loss_fn_1_28379P
Llstm_lstm_lstm_cell_recurrent_kernel_regularizer_abs_readvariableop_resource
identity
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpLlstm_lstm_lstm_cell_recurrent_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add{
IdentityIdentity8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Č

)__inference_lstm_lstm_layer_call_fn_28100

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*+
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_262752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ś
š
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27327
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimØ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimŖ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterŁ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_27189*
condR
while_cond_27188*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeŽ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addģ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::2
whilewhile:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ī
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_26275

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimØ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimŖ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterŁ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_26137*
condR
while_cond_26136*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeŽ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addģ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
“
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_28106

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ö
while_cond_27696
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_27696___redundant_placeholder0-
)while_cond_27696___redundant_placeholder1-
)while_cond_27696___redundant_placeholder2-
)while_cond_27696___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

ö
while_cond_25893
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_25893___redundant_placeholder0-
)while_cond_25893___redundant_placeholder1-
)while_cond_25893___redundant_placeholder2-
)while_cond_25893___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ŗ	

"model_1_lstm_lstm_while_cond_24874(
$model_1_lstm_lstm_while_loop_counter.
*model_1_lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3*
&less_model_1_lstm_lstm_strided_slice_1?
;model_1_lstm_lstm_while_cond_24874___redundant_placeholder0?
;model_1_lstm_lstm_while_cond_24874___redundant_placeholder1?
;model_1_lstm_lstm_while_cond_24874___redundant_placeholder2?
;model_1_lstm_lstm_while_cond_24874___redundant_placeholder3
identity
j
LessLessplaceholder&less_model_1_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Č
v
__inference_loss_fn_0_28366F
Blstm_lstm_lstm_cell_kernel_regularizer_abs_readvariableop_resource
identitył
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBlstm_lstm_lstm_cell_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addq
IdentityIdentity.lstm_lstm/lstm_cell/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 

ö
while_cond_25691
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_25691___redundant_placeholder0-
)while_cond_25691___redundant_placeholder1-
)while_cond_25691___redundant_placeholder2-
)while_cond_25691___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
`

lstm_lstm_while_body_26891 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_lstm_strided_slice_1_0[
Wtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_lstm_strided_slice_1Y
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeæ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yh
add_1AddV2lstm_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityi

Identity_1Identity"lstm_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"8
lstm_lstm_strided_slice_1lstm_lstm_strided_slice_1_0"°
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
£l
Å
!__inference__traced_restore_28562
file_prefix&
"assignvariableop_rnn_densef_kernel&
"assignvariableop_1_rnn_densef_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate1
-assignvariableop_7_lstm_lstm_lstm_cell_kernel;
7assignvariableop_8_lstm_lstm_lstm_cell_recurrent_kernel/
+assignvariableop_9_lstm_lstm_lstm_cell_bias
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_10
,assignvariableop_14_adam_rnn_densef_kernel_m.
*assignvariableop_15_adam_rnn_densef_bias_m9
5assignvariableop_16_adam_lstm_lstm_lstm_cell_kernel_mC
?assignvariableop_17_adam_lstm_lstm_lstm_cell_recurrent_kernel_m7
3assignvariableop_18_adam_lstm_lstm_lstm_cell_bias_m0
,assignvariableop_19_adam_rnn_densef_kernel_v.
*assignvariableop_20_adam_rnn_densef_bias_v9
5assignvariableop_21_adam_lstm_lstm_lstm_cell_kernel_vC
?assignvariableop_22_adam_lstm_lstm_lstm_cell_recurrent_kernel_v7
3assignvariableop_23_adam_lstm_lstm_lstm_cell_bias_v
identity_25¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names¾
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices£
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp"assignvariableop_rnn_densef_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp"assignvariableop_1_rnn_densef_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOp-assignvariableop_7_lstm_lstm_lstm_cell_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8­
AssignVariableOp_8AssignVariableOp7assignvariableop_8_lstm_lstm_lstm_cell_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9”
AssignVariableOp_9AssignVariableOp+assignvariableop_9_lstm_lstm_lstm_cell_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14„
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_rnn_densef_kernel_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_rnn_densef_bias_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adam_lstm_lstm_lstm_cell_kernel_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17ø
AssignVariableOp_17AssignVariableOp?assignvariableop_17_adam_lstm_lstm_lstm_cell_recurrent_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18¬
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_lstm_lstm_lstm_cell_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19„
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_rnn_densef_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_rnn_densef_bias_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_lstm_lstm_lstm_cell_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22ø
AssignVariableOp_22AssignVariableOp?assignvariableop_22_adam_lstm_lstm_lstm_cell_recurrent_kernel_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23¬
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_lstm_lstm_lstm_cell_bias_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23Ø
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpī
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24ū
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ö
while_cond_27188
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_27188___redundant_placeholder0-
)while_cond_27188___redundant_placeholder1-
)while_cond_27188___redundant_placeholder2-
)while_cond_27188___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ų+

B__inference_model_1_layer_call_and_return_conditional_losses_26480

inputs
lstm_lstm_26450
lstm_lstm_26452
lstm_lstm_26454
rnn_densef_26458
rnn_densef_26460
identity¢!lstm_lstm/StatefulPartitionedCall¢"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_26450lstm_lstm_26452lstm_lstm_26454*
Tin
2*
Tout
2*+
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_262752#
!lstm_lstm/StatefulPartitionedCallÜ
flatten_1/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_263112
flatten_1/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0rnn_densef_26458rnn_densef_26460*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_rnn_densef_layer_call_and_return_conditional_losses_263302$
"rnn_densef/StatefulPartitionedCallĘ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26450*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addŚ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26454*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addČ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ž^
Ļ
while_body_27189
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
Ś
š
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27570
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimØ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimŖ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterŁ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_27432*
condR
while_cond_27431*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeŽ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addģ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::2
whilewhile:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ń
­
E__inference_rnn_densef_layer_call_and_return_conditional_losses_26330

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ą*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’Ą:::P L
(
_output_shapes
:’’’’’’’’’Ą
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

¦
#__inference_signature_wrapper_26534
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_250062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ö
while_cond_27939
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_27939___redundant_placeholder0-
)while_cond_27939___redundant_placeholder1-
)while_cond_27939___redundant_placeholder2-
)while_cond_27939___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ž^
Ļ
while_body_27432
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ā
Ę
lstm_lstm_while_cond_26638 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
less_lstm_lstm_strided_slice_17
3lstm_lstm_while_cond_26638___redundant_placeholder07
3lstm_lstm_while_cond_26638___redundant_placeholder17
3lstm_lstm_while_cond_26638___redundant_placeholder27
3lstm_lstm_while_cond_26638___redundant_placeholder3
identity
b
LessLessplaceholderless_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ÆÄ
ó
B__inference_model_1_layer_call_and_return_conditional_losses_26786

inputs5
1lstm_lstm_lstm_cell_split_readvariableop_resource7
3lstm_lstm_lstm_cell_split_1_readvariableop_resource/
+lstm_lstm_lstm_cell_readvariableop_resource-
)rnn_densef_matmul_readvariableop_resource.
*rnn_densef_biasadd_readvariableop_resource
identity¢lstm_lstm/whileX
lstm_lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_lstm/Shape
lstm_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_lstm/strided_slice/stack
lstm_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_1
lstm_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_2
lstm_lstm/strided_sliceStridedSlicelstm_lstm/Shape:output:0&lstm_lstm/strided_slice/stack:output:0(lstm_lstm/strided_slice/stack_1:output:0(lstm_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slicep
lstm_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/mul/y
lstm_lstm/zeros/mulMul lstm_lstm/strided_slice:output:0lstm_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/muls
lstm_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
lstm_lstm/zeros/Less/y
lstm_lstm/zeros/LessLesslstm_lstm/zeros/mul:z:0lstm_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/Lessv
lstm_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/packed/1«
lstm_lstm/zeros/packedPack lstm_lstm/strided_slice:output:0!lstm_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros/packeds
lstm_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros/Const
lstm_lstm/zerosFilllstm_lstm/zeros/packed:output:0lstm_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/zerost
lstm_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/mul/y
lstm_lstm/zeros_1/mulMul lstm_lstm/strided_slice:output:0 lstm_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/mulw
lstm_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
lstm_lstm/zeros_1/Less/y
lstm_lstm/zeros_1/LessLesslstm_lstm/zeros_1/mul:z:0!lstm_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/Lessz
lstm_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/packed/1±
lstm_lstm/zeros_1/packedPack lstm_lstm/strided_slice:output:0#lstm_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros_1/packedw
lstm_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros_1/Const„
lstm_lstm/zeros_1Fill!lstm_lstm/zeros_1/packed:output:0 lstm_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/zeros_1
lstm_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose/perm
lstm_lstm/transpose	Transposeinputs!lstm_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
lstm_lstm/transposem
lstm_lstm/Shape_1Shapelstm_lstm/transpose:y:0*
T0*
_output_shapes
:2
lstm_lstm/Shape_1
lstm_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_1/stack
!lstm_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_1
!lstm_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_2Ŗ
lstm_lstm/strided_slice_1StridedSlicelstm_lstm/Shape_1:output:0(lstm_lstm/strided_slice_1/stack:output:0*lstm_lstm/strided_slice_1/stack_1:output:0*lstm_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slice_1
%lstm_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%lstm_lstm/TensorArrayV2/element_shapeŚ
lstm_lstm/TensorArrayV2TensorListReserve.lstm_lstm/TensorArrayV2/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2Ó
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2A
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape 
1lstm_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_lstm/transpose:y:0Hlstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1lstm_lstm/TensorArrayUnstack/TensorListFromTensor
lstm_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_2/stack
!lstm_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_1
!lstm_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_2ø
lstm_lstm/strided_slice_2StridedSlicelstm_lstm/transpose:y:0(lstm_lstm/strided_slice_2/stack:output:0*lstm_lstm/strided_slice_2/stack_1:output:0*lstm_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
lstm_lstm/strided_slice_2x
lstm_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const
#lstm_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_lstm/lstm_cell/split/split_dimĘ
(lstm_lstm/lstm_cell/split/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02*
(lstm_lstm/lstm_cell/split/ReadVariableOp÷
lstm_lstm/lstm_cell/splitSplit,lstm_lstm/lstm_cell/split/split_dim:output:00lstm_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_lstm/lstm_cell/split¼
lstm_lstm/lstm_cell/MatMulMatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMulĄ
lstm_lstm/lstm_cell/MatMul_1MatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_1Ą
lstm_lstm/lstm_cell/MatMul_2MatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_2Ą
lstm_lstm/lstm_cell/MatMul_3MatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_3|
lstm_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const_1
%lstm_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_lstm/lstm_cell/split_1/split_dimČ
*lstm_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02,
*lstm_lstm/lstm_cell/split_1/ReadVariableOpļ
lstm_lstm/lstm_cell/split_1Split.lstm_lstm/lstm_cell/split_1/split_dim:output:02lstm_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_lstm/lstm_cell/split_1Ć
lstm_lstm/lstm_cell/BiasAddBiasAdd$lstm_lstm/lstm_cell/MatMul:product:0$lstm_lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAddÉ
lstm_lstm/lstm_cell/BiasAdd_1BiasAdd&lstm_lstm/lstm_cell/MatMul_1:product:0$lstm_lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAdd_1É
lstm_lstm/lstm_cell/BiasAdd_2BiasAdd&lstm_lstm/lstm_cell/MatMul_2:product:0$lstm_lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAdd_2É
lstm_lstm/lstm_cell/BiasAdd_3BiasAdd&lstm_lstm/lstm_cell/MatMul_3:product:0$lstm_lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAdd_3“
"lstm_lstm/lstm_cell/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02$
"lstm_lstm/lstm_cell/ReadVariableOp£
'lstm_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_lstm/lstm_cell/strided_slice/stack§
)lstm_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice/stack_1§
)lstm_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_lstm/lstm_cell/strided_slice/stack_2ō
!lstm_lstm/lstm_cell/strided_sliceStridedSlice*lstm_lstm/lstm_cell/ReadVariableOp:value:00lstm_lstm/lstm_cell/strided_slice/stack:output:02lstm_lstm/lstm_cell/strided_slice/stack_1:output:02lstm_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!lstm_lstm/lstm_cell/strided_slice¾
lstm_lstm/lstm_cell/MatMul_4MatMullstm_lstm/zeros:output:0*lstm_lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_4»
lstm_lstm/lstm_cell/addAddV2$lstm_lstm/lstm_cell/BiasAdd:output:0&lstm_lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add
lstm_lstm/lstm_cell/SigmoidSigmoidlstm_lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Sigmoidø
$lstm_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_1§
)lstm_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice_1/stack«
+lstm_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_1«
+lstm_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_2
#lstm_lstm/lstm_cell/strided_slice_1StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_1:value:02lstm_lstm/lstm_cell/strided_slice_1/stack:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_1Ą
lstm_lstm/lstm_cell/MatMul_5MatMullstm_lstm/zeros:output:0,lstm_lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_5Į
lstm_lstm/lstm_cell/add_1AddV2&lstm_lstm/lstm_cell/BiasAdd_1:output:0&lstm_lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_1
lstm_lstm/lstm_cell/Sigmoid_1Sigmoidlstm_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Sigmoid_1Ŗ
lstm_lstm/lstm_cell/mulMul!lstm_lstm/lstm_cell/Sigmoid_1:y:0lstm_lstm/zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/mulø
$lstm_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_2§
)lstm_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_lstm/lstm_cell/strided_slice_2/stack«
+lstm_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_1«
+lstm_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_2
#lstm_lstm/lstm_cell/strided_slice_2StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_2:value:02lstm_lstm/lstm_cell/strided_slice_2/stack:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_2Ą
lstm_lstm/lstm_cell/MatMul_6MatMullstm_lstm/zeros:output:0,lstm_lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_6Į
lstm_lstm/lstm_cell/add_2AddV2&lstm_lstm/lstm_cell/BiasAdd_2:output:0&lstm_lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_2
lstm_lstm/lstm_cell/ReluRelulstm_lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Reluø
lstm_lstm/lstm_cell/mul_1Mullstm_lstm/lstm_cell/Sigmoid:y:0&lstm_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/mul_1­
lstm_lstm/lstm_cell/add_3AddV2lstm_lstm/lstm_cell/mul:z:0lstm_lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_3ø
$lstm_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_3§
)lstm_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2+
)lstm_lstm/lstm_cell/strided_slice_3/stack«
+lstm_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_1«
+lstm_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_2
#lstm_lstm/lstm_cell/strided_slice_3StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_3:value:02lstm_lstm/lstm_cell/strided_slice_3/stack:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_3Ą
lstm_lstm/lstm_cell/MatMul_7MatMullstm_lstm/zeros:output:0,lstm_lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_7Į
lstm_lstm/lstm_cell/add_4AddV2&lstm_lstm/lstm_cell/BiasAdd_3:output:0&lstm_lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_4
lstm_lstm/lstm_cell/Sigmoid_2Sigmoidlstm_lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Sigmoid_2
lstm_lstm/lstm_cell/Relu_1Relulstm_lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Relu_1¼
lstm_lstm/lstm_cell/mul_2Mul!lstm_lstm/lstm_cell/Sigmoid_2:y:0(lstm_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/mul_2£
'lstm_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2)
'lstm_lstm/TensorArrayV2_1/element_shapeą
lstm_lstm/TensorArrayV2_1TensorListReserve0lstm_lstm/TensorArrayV2_1/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2_1b
lstm_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/time
"lstm_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2$
"lstm_lstm/while/maximum_iterations~
lstm_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/while/loop_counterļ
lstm_lstm/whileWhile%lstm_lstm/while/loop_counter:output:0+lstm_lstm/while/maximum_iterations:output:0lstm_lstm/time:output:0"lstm_lstm/TensorArrayV2_1:handle:0lstm_lstm/zeros:output:0lstm_lstm/zeros_1:output:0"lstm_lstm/strided_slice_1:output:0Alstm_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_lstm_lstm_cell_split_readvariableop_resource3lstm_lstm_lstm_cell_split_1_readvariableop_resource+lstm_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*&
bodyR
lstm_lstm_while_body_26639*&
condR
lstm_lstm_while_cond_26638*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
lstm_lstm/whileÉ
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2<
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape
,lstm_lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm_lstm/while:output:3Clstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’*
element_dtype02.
,lstm_lstm/TensorArrayV2Stack/TensorListStack
lstm_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2!
lstm_lstm/strided_slice_3/stack
!lstm_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!lstm_lstm/strided_slice_3/stack_1
!lstm_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_3/stack_2Ö
lstm_lstm/strided_slice_3StridedSlice5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0(lstm_lstm/strided_slice_3/stack:output:0*lstm_lstm/strided_slice_3/stack_1:output:0*lstm_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
lstm_lstm/strided_slice_3
lstm_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose_1/permĶ
lstm_lstm/transpose_1	Transpose5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
lstm_lstm/transpose_1z
lstm_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/runtimes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’@  2
flatten_1/Const
flatten_1/ReshapeReshapelstm_lstm/transpose_1:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2
flatten_1/ReshapeÆ
 rnn_densef/MatMul/ReadVariableOpReadVariableOp)rnn_densef_matmul_readvariableop_resource*
_output_shapes
:	Ą*
dtype02"
 rnn_densef/MatMul/ReadVariableOpØ
rnn_densef/MatMulMatMulflatten_1/Reshape:output:0(rnn_densef/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rnn_densef/MatMul­
!rnn_densef/BiasAdd/ReadVariableOpReadVariableOp*rnn_densef_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rnn_densef/BiasAdd/ReadVariableOp­
rnn_densef/BiasAddBiasAddrnn_densef/MatMul:product:0)rnn_densef/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rnn_densef/BiasAdd
rnn_densef/SoftmaxSoftmaxrnn_densef/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
rnn_densef/Softmaxč
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addö
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentityrnn_densef/Softmax:softmax:0^lstm_lstm/while*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::2"
lstm_lstm/whilelstm_lstm/while:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ėX

D__inference_lstm_cell_layer_call_and_return_conditional_losses_25218

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOp§
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicep
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1r
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:’’’’’’’’’2
mul|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2r
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_1]
add_3AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3r
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_2Ō
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addā
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ķ_
Ī
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_25629

inputs
lstm_cell_25531
lstm_cell_25533
lstm_cell_25535
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2č
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25531lstm_cell_25533lstm_cell_25535*
Tin

2*
Tout
2*M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_251322#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25531lstm_cell_25533lstm_cell_25535*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_25544*
condR
while_cond_25543*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeĘ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_25531*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addŚ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_25535*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ń
­
E__inference_rnn_densef_layer_call_and_return_conditional_losses_28122

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ą*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’Ą:::P L
(
_output_shapes
:’’’’’’’’’Ą
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ÆÄ
ó
B__inference_model_1_layer_call_and_return_conditional_losses_27038

inputs5
1lstm_lstm_lstm_cell_split_readvariableop_resource7
3lstm_lstm_lstm_cell_split_1_readvariableop_resource/
+lstm_lstm_lstm_cell_readvariableop_resource-
)rnn_densef_matmul_readvariableop_resource.
*rnn_densef_biasadd_readvariableop_resource
identity¢lstm_lstm/whileX
lstm_lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_lstm/Shape
lstm_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_lstm/strided_slice/stack
lstm_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_1
lstm_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_2
lstm_lstm/strided_sliceStridedSlicelstm_lstm/Shape:output:0&lstm_lstm/strided_slice/stack:output:0(lstm_lstm/strided_slice/stack_1:output:0(lstm_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slicep
lstm_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/mul/y
lstm_lstm/zeros/mulMul lstm_lstm/strided_slice:output:0lstm_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/muls
lstm_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
lstm_lstm/zeros/Less/y
lstm_lstm/zeros/LessLesslstm_lstm/zeros/mul:z:0lstm_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/Lessv
lstm_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/packed/1«
lstm_lstm/zeros/packedPack lstm_lstm/strided_slice:output:0!lstm_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros/packeds
lstm_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros/Const
lstm_lstm/zerosFilllstm_lstm/zeros/packed:output:0lstm_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/zerost
lstm_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/mul/y
lstm_lstm/zeros_1/mulMul lstm_lstm/strided_slice:output:0 lstm_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/mulw
lstm_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
lstm_lstm/zeros_1/Less/y
lstm_lstm/zeros_1/LessLesslstm_lstm/zeros_1/mul:z:0!lstm_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/Lessz
lstm_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/packed/1±
lstm_lstm/zeros_1/packedPack lstm_lstm/strided_slice:output:0#lstm_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros_1/packedw
lstm_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros_1/Const„
lstm_lstm/zeros_1Fill!lstm_lstm/zeros_1/packed:output:0 lstm_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/zeros_1
lstm_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose/perm
lstm_lstm/transpose	Transposeinputs!lstm_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
lstm_lstm/transposem
lstm_lstm/Shape_1Shapelstm_lstm/transpose:y:0*
T0*
_output_shapes
:2
lstm_lstm/Shape_1
lstm_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_1/stack
!lstm_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_1
!lstm_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_2Ŗ
lstm_lstm/strided_slice_1StridedSlicelstm_lstm/Shape_1:output:0(lstm_lstm/strided_slice_1/stack:output:0*lstm_lstm/strided_slice_1/stack_1:output:0*lstm_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slice_1
%lstm_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%lstm_lstm/TensorArrayV2/element_shapeŚ
lstm_lstm/TensorArrayV2TensorListReserve.lstm_lstm/TensorArrayV2/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2Ó
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2A
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape 
1lstm_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_lstm/transpose:y:0Hlstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1lstm_lstm/TensorArrayUnstack/TensorListFromTensor
lstm_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_2/stack
!lstm_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_1
!lstm_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_2ø
lstm_lstm/strided_slice_2StridedSlicelstm_lstm/transpose:y:0(lstm_lstm/strided_slice_2/stack:output:0*lstm_lstm/strided_slice_2/stack_1:output:0*lstm_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
lstm_lstm/strided_slice_2x
lstm_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const
#lstm_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_lstm/lstm_cell/split/split_dimĘ
(lstm_lstm/lstm_cell/split/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02*
(lstm_lstm/lstm_cell/split/ReadVariableOp÷
lstm_lstm/lstm_cell/splitSplit,lstm_lstm/lstm_cell/split/split_dim:output:00lstm_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_lstm/lstm_cell/split¼
lstm_lstm/lstm_cell/MatMulMatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMulĄ
lstm_lstm/lstm_cell/MatMul_1MatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_1Ą
lstm_lstm/lstm_cell/MatMul_2MatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_2Ą
lstm_lstm/lstm_cell/MatMul_3MatMul"lstm_lstm/strided_slice_2:output:0"lstm_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_3|
lstm_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const_1
%lstm_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_lstm/lstm_cell/split_1/split_dimČ
*lstm_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02,
*lstm_lstm/lstm_cell/split_1/ReadVariableOpļ
lstm_lstm/lstm_cell/split_1Split.lstm_lstm/lstm_cell/split_1/split_dim:output:02lstm_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_lstm/lstm_cell/split_1Ć
lstm_lstm/lstm_cell/BiasAddBiasAdd$lstm_lstm/lstm_cell/MatMul:product:0$lstm_lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAddÉ
lstm_lstm/lstm_cell/BiasAdd_1BiasAdd&lstm_lstm/lstm_cell/MatMul_1:product:0$lstm_lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAdd_1É
lstm_lstm/lstm_cell/BiasAdd_2BiasAdd&lstm_lstm/lstm_cell/MatMul_2:product:0$lstm_lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAdd_2É
lstm_lstm/lstm_cell/BiasAdd_3BiasAdd&lstm_lstm/lstm_cell/MatMul_3:product:0$lstm_lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/BiasAdd_3“
"lstm_lstm/lstm_cell/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02$
"lstm_lstm/lstm_cell/ReadVariableOp£
'lstm_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_lstm/lstm_cell/strided_slice/stack§
)lstm_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice/stack_1§
)lstm_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_lstm/lstm_cell/strided_slice/stack_2ō
!lstm_lstm/lstm_cell/strided_sliceStridedSlice*lstm_lstm/lstm_cell/ReadVariableOp:value:00lstm_lstm/lstm_cell/strided_slice/stack:output:02lstm_lstm/lstm_cell/strided_slice/stack_1:output:02lstm_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!lstm_lstm/lstm_cell/strided_slice¾
lstm_lstm/lstm_cell/MatMul_4MatMullstm_lstm/zeros:output:0*lstm_lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_4»
lstm_lstm/lstm_cell/addAddV2$lstm_lstm/lstm_cell/BiasAdd:output:0&lstm_lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add
lstm_lstm/lstm_cell/SigmoidSigmoidlstm_lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Sigmoidø
$lstm_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_1§
)lstm_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice_1/stack«
+lstm_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_1«
+lstm_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_2
#lstm_lstm/lstm_cell/strided_slice_1StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_1:value:02lstm_lstm/lstm_cell/strided_slice_1/stack:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_1Ą
lstm_lstm/lstm_cell/MatMul_5MatMullstm_lstm/zeros:output:0,lstm_lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_5Į
lstm_lstm/lstm_cell/add_1AddV2&lstm_lstm/lstm_cell/BiasAdd_1:output:0&lstm_lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_1
lstm_lstm/lstm_cell/Sigmoid_1Sigmoidlstm_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Sigmoid_1Ŗ
lstm_lstm/lstm_cell/mulMul!lstm_lstm/lstm_cell/Sigmoid_1:y:0lstm_lstm/zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/mulø
$lstm_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_2§
)lstm_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_lstm/lstm_cell/strided_slice_2/stack«
+lstm_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_1«
+lstm_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_2
#lstm_lstm/lstm_cell/strided_slice_2StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_2:value:02lstm_lstm/lstm_cell/strided_slice_2/stack:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_2Ą
lstm_lstm/lstm_cell/MatMul_6MatMullstm_lstm/zeros:output:0,lstm_lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_6Į
lstm_lstm/lstm_cell/add_2AddV2&lstm_lstm/lstm_cell/BiasAdd_2:output:0&lstm_lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_2
lstm_lstm/lstm_cell/ReluRelulstm_lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Reluø
lstm_lstm/lstm_cell/mul_1Mullstm_lstm/lstm_cell/Sigmoid:y:0&lstm_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/mul_1­
lstm_lstm/lstm_cell/add_3AddV2lstm_lstm/lstm_cell/mul:z:0lstm_lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_3ø
$lstm_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_3§
)lstm_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2+
)lstm_lstm/lstm_cell/strided_slice_3/stack«
+lstm_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_1«
+lstm_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_2
#lstm_lstm/lstm_cell/strided_slice_3StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_3:value:02lstm_lstm/lstm_cell/strided_slice_3/stack:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_3Ą
lstm_lstm/lstm_cell/MatMul_7MatMullstm_lstm/zeros:output:0,lstm_lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/MatMul_7Į
lstm_lstm/lstm_cell/add_4AddV2&lstm_lstm/lstm_cell/BiasAdd_3:output:0&lstm_lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/add_4
lstm_lstm/lstm_cell/Sigmoid_2Sigmoidlstm_lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Sigmoid_2
lstm_lstm/lstm_cell/Relu_1Relulstm_lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/Relu_1¼
lstm_lstm/lstm_cell/mul_2Mul!lstm_lstm/lstm_cell/Sigmoid_2:y:0(lstm_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_lstm/lstm_cell/mul_2£
'lstm_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2)
'lstm_lstm/TensorArrayV2_1/element_shapeą
lstm_lstm/TensorArrayV2_1TensorListReserve0lstm_lstm/TensorArrayV2_1/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2_1b
lstm_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/time
"lstm_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2$
"lstm_lstm/while/maximum_iterations~
lstm_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/while/loop_counterļ
lstm_lstm/whileWhile%lstm_lstm/while/loop_counter:output:0+lstm_lstm/while/maximum_iterations:output:0lstm_lstm/time:output:0"lstm_lstm/TensorArrayV2_1:handle:0lstm_lstm/zeros:output:0lstm_lstm/zeros_1:output:0"lstm_lstm/strided_slice_1:output:0Alstm_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_lstm_lstm_cell_split_readvariableop_resource3lstm_lstm_lstm_cell_split_1_readvariableop_resource+lstm_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*&
bodyR
lstm_lstm_while_body_26891*&
condR
lstm_lstm_while_cond_26890*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
lstm_lstm/whileÉ
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2<
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape
,lstm_lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm_lstm/while:output:3Clstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’*
element_dtype02.
,lstm_lstm/TensorArrayV2Stack/TensorListStack
lstm_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2!
lstm_lstm/strided_slice_3/stack
!lstm_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!lstm_lstm/strided_slice_3/stack_1
!lstm_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_3/stack_2Ö
lstm_lstm/strided_slice_3StridedSlice5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0(lstm_lstm/strided_slice_3/stack:output:0*lstm_lstm/strided_slice_3/stack_1:output:0*lstm_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
lstm_lstm/strided_slice_3
lstm_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose_1/permĶ
lstm_lstm/transpose_1	Transpose5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
lstm_lstm/transpose_1z
lstm_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/runtimes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’@  2
flatten_1/Const
flatten_1/ReshapeReshapelstm_lstm/transpose_1:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2
flatten_1/ReshapeÆ
 rnn_densef/MatMul/ReadVariableOpReadVariableOp)rnn_densef_matmul_readvariableop_resource*
_output_shapes
:	Ą*
dtype02"
 rnn_densef/MatMul/ReadVariableOpØ
rnn_densef/MatMulMatMulflatten_1/Reshape:output:0(rnn_densef/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rnn_densef/MatMul­
!rnn_densef/BiasAdd/ReadVariableOpReadVariableOp*rnn_densef_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rnn_densef/BiasAdd/ReadVariableOp­
rnn_densef/BiasAddBiasAddrnn_densef/MatMul:product:0)rnn_densef/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rnn_densef/BiasAdd
rnn_densef/SoftmaxSoftmaxrnn_densef/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
rnn_densef/Softmaxč
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addö
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentityrnn_densef/Softmax:softmax:0^lstm_lstm/while*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::2"
lstm_lstm/whilelstm_lstm/while:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ś`
Ķ
"model_1_lstm_lstm_while_body_24875(
$model_1_lstm_lstm_while_loop_counter.
*model_1_lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#model_1_lstm_lstm_strided_slice_1_0c
_tensorarrayv2read_tensorlistgetitem_model_1_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5%
!model_1_lstm_lstm_strided_slice_1a
]tensorarrayv2read_tensorlistgetitem_model_1_lstm_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeĒ
#TensorArrayV2Read/TensorListGetItemTensorListGetItem_tensorarrayv2read_tensorlistgetitem_model_1_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yp
add_1AddV2$model_1_lstm_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityq

Identity_1Identity*model_1_lstm_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"H
!model_1_lstm_lstm_strided_slice_1#model_1_lstm_lstm_strided_slice_1_0"Ą
]tensorarrayv2read_tensorlistgetitem_model_1_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_tensorarrayv2read_tensorlistgetitem_model_1_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
»
Ŗ
'__inference_model_1_layer_call_fn_26445
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallū
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_264322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ö
while_cond_25543
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_25543___redundant_placeholder0-
)while_cond_25543___redundant_placeholder1-
)while_cond_25543___redundant_placeholder2-
)while_cond_25543___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ūæ

 __inference__wrapped_model_25006
input_1=
9model_1_lstm_lstm_lstm_cell_split_readvariableop_resource?
;model_1_lstm_lstm_lstm_cell_split_1_readvariableop_resource7
3model_1_lstm_lstm_lstm_cell_readvariableop_resource5
1model_1_rnn_densef_matmul_readvariableop_resource6
2model_1_rnn_densef_biasadd_readvariableop_resource
identity¢model_1/lstm_lstm/whilei
model_1/lstm_lstm/ShapeShapeinput_1*
T0*
_output_shapes
:2
model_1/lstm_lstm/Shape
%model_1/lstm_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/lstm_lstm/strided_slice/stack
'model_1/lstm_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/lstm_lstm/strided_slice/stack_1
'model_1/lstm_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/lstm_lstm/strided_slice/stack_2Ī
model_1/lstm_lstm/strided_sliceStridedSlice model_1/lstm_lstm/Shape:output:0.model_1/lstm_lstm/strided_slice/stack:output:00model_1/lstm_lstm/strided_slice/stack_1:output:00model_1/lstm_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_1/lstm_lstm/strided_slice
model_1/lstm_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/lstm_lstm/zeros/mul/y“
model_1/lstm_lstm/zeros/mulMul(model_1/lstm_lstm/strided_slice:output:0&model_1/lstm_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_lstm/zeros/mul
model_1/lstm_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2 
model_1/lstm_lstm/zeros/Less/yÆ
model_1/lstm_lstm/zeros/LessLessmodel_1/lstm_lstm/zeros/mul:z:0'model_1/lstm_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_lstm/zeros/Less
 model_1/lstm_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 model_1/lstm_lstm/zeros/packed/1Ė
model_1/lstm_lstm/zeros/packedPack(model_1/lstm_lstm/strided_slice:output:0)model_1/lstm_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_1/lstm_lstm/zeros/packed
model_1/lstm_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/lstm_lstm/zeros/Const½
model_1/lstm_lstm/zerosFill'model_1/lstm_lstm/zeros/packed:output:0&model_1/lstm_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
model_1/lstm_lstm/zeros
model_1/lstm_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/lstm_lstm/zeros_1/mul/yŗ
model_1/lstm_lstm/zeros_1/mulMul(model_1/lstm_lstm/strided_slice:output:0(model_1/lstm_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_lstm/zeros_1/mul
 model_1/lstm_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2"
 model_1/lstm_lstm/zeros_1/Less/y·
model_1/lstm_lstm/zeros_1/LessLess!model_1/lstm_lstm/zeros_1/mul:z:0)model_1/lstm_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
model_1/lstm_lstm/zeros_1/Less
"model_1/lstm_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/lstm_lstm/zeros_1/packed/1Ń
 model_1/lstm_lstm/zeros_1/packedPack(model_1/lstm_lstm/strided_slice:output:0+model_1/lstm_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 model_1/lstm_lstm/zeros_1/packed
model_1/lstm_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model_1/lstm_lstm/zeros_1/ConstÅ
model_1/lstm_lstm/zeros_1Fill)model_1/lstm_lstm/zeros_1/packed:output:0(model_1/lstm_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
model_1/lstm_lstm/zeros_1
 model_1/lstm_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_1/lstm_lstm/transpose/perm±
model_1/lstm_lstm/transpose	Transposeinput_1)model_1/lstm_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
model_1/lstm_lstm/transpose
model_1/lstm_lstm/Shape_1Shapemodel_1/lstm_lstm/transpose:y:0*
T0*
_output_shapes
:2
model_1/lstm_lstm/Shape_1
'model_1/lstm_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_1/lstm_lstm/strided_slice_1/stack 
)model_1/lstm_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_1/lstm_lstm/strided_slice_1/stack_1 
)model_1/lstm_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_1/lstm_lstm/strided_slice_1/stack_2Ś
!model_1/lstm_lstm/strided_slice_1StridedSlice"model_1/lstm_lstm/Shape_1:output:00model_1/lstm_lstm/strided_slice_1/stack:output:02model_1/lstm_lstm/strided_slice_1/stack_1:output:02model_1/lstm_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_1/lstm_lstm/strided_slice_1©
-model_1/lstm_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2/
-model_1/lstm_lstm/TensorArrayV2/element_shapeś
model_1/lstm_lstm/TensorArrayV2TensorListReserve6model_1/lstm_lstm/TensorArrayV2/element_shape:output:0*model_1/lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model_1/lstm_lstm/TensorArrayV2ć
Gmodel_1/lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2I
Gmodel_1/lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeĄ
9model_1/lstm_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_1/lstm_lstm/transpose:y:0Pmodel_1/lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02;
9model_1/lstm_lstm/TensorArrayUnstack/TensorListFromTensor
'model_1/lstm_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_1/lstm_lstm/strided_slice_2/stack 
)model_1/lstm_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_1/lstm_lstm/strided_slice_2/stack_1 
)model_1/lstm_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_1/lstm_lstm/strided_slice_2/stack_2č
!model_1/lstm_lstm/strided_slice_2StridedSlicemodel_1/lstm_lstm/transpose:y:00model_1/lstm_lstm/strided_slice_2/stack:output:02model_1/lstm_lstm/strided_slice_2/stack_1:output:02model_1/lstm_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2#
!model_1/lstm_lstm/strided_slice_2
!model_1/lstm_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/lstm_lstm/lstm_cell/Const
+model_1/lstm_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_1/lstm_lstm/lstm_cell/split/split_dimŽ
0model_1/lstm_lstm/lstm_cell/split/ReadVariableOpReadVariableOp9model_1_lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype022
0model_1/lstm_lstm/lstm_cell/split/ReadVariableOp
!model_1/lstm_lstm/lstm_cell/splitSplit4model_1/lstm_lstm/lstm_cell/split/split_dim:output:08model_1/lstm_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2#
!model_1/lstm_lstm/lstm_cell/splitÜ
"model_1/lstm_lstm/lstm_cell/MatMulMatMul*model_1/lstm_lstm/strided_slice_2:output:0*model_1/lstm_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2$
"model_1/lstm_lstm/lstm_cell/MatMulą
$model_1/lstm_lstm/lstm_cell/MatMul_1MatMul*model_1/lstm_lstm/strided_slice_2:output:0*model_1/lstm_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2&
$model_1/lstm_lstm/lstm_cell/MatMul_1ą
$model_1/lstm_lstm/lstm_cell/MatMul_2MatMul*model_1/lstm_lstm/strided_slice_2:output:0*model_1/lstm_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2&
$model_1/lstm_lstm/lstm_cell/MatMul_2ą
$model_1/lstm_lstm/lstm_cell/MatMul_3MatMul*model_1/lstm_lstm/strided_slice_2:output:0*model_1/lstm_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2&
$model_1/lstm_lstm/lstm_cell/MatMul_3
#model_1/lstm_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_1/lstm_lstm/lstm_cell/Const_1 
-model_1/lstm_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_1/lstm_lstm/lstm_cell/split_1/split_dimą
2model_1/lstm_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp;model_1_lstm_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_1/lstm_lstm/lstm_cell/split_1/ReadVariableOp
#model_1/lstm_lstm/lstm_cell/split_1Split6model_1/lstm_lstm/lstm_cell/split_1/split_dim:output:0:model_1/lstm_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2%
#model_1/lstm_lstm/lstm_cell/split_1ć
#model_1/lstm_lstm/lstm_cell/BiasAddBiasAdd,model_1/lstm_lstm/lstm_cell/MatMul:product:0,model_1/lstm_lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#model_1/lstm_lstm/lstm_cell/BiasAddé
%model_1/lstm_lstm/lstm_cell/BiasAdd_1BiasAdd.model_1/lstm_lstm/lstm_cell/MatMul_1:product:0,model_1/lstm_lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2'
%model_1/lstm_lstm/lstm_cell/BiasAdd_1é
%model_1/lstm_lstm/lstm_cell/BiasAdd_2BiasAdd.model_1/lstm_lstm/lstm_cell/MatMul_2:product:0,model_1/lstm_lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2'
%model_1/lstm_lstm/lstm_cell/BiasAdd_2é
%model_1/lstm_lstm/lstm_cell/BiasAdd_3BiasAdd.model_1/lstm_lstm/lstm_cell/MatMul_3:product:0,model_1/lstm_lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2'
%model_1/lstm_lstm/lstm_cell/BiasAdd_3Ģ
*model_1/lstm_lstm/lstm_cell/ReadVariableOpReadVariableOp3model_1_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02,
*model_1/lstm_lstm/lstm_cell/ReadVariableOp³
/model_1/lstm_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/model_1/lstm_lstm/lstm_cell/strided_slice/stack·
1model_1/lstm_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1model_1/lstm_lstm/lstm_cell/strided_slice/stack_1·
1model_1/lstm_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1model_1/lstm_lstm/lstm_cell/strided_slice/stack_2¤
)model_1/lstm_lstm/lstm_cell/strided_sliceStridedSlice2model_1/lstm_lstm/lstm_cell/ReadVariableOp:value:08model_1/lstm_lstm/lstm_cell/strided_slice/stack:output:0:model_1/lstm_lstm/lstm_cell/strided_slice/stack_1:output:0:model_1/lstm_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2+
)model_1/lstm_lstm/lstm_cell/strided_sliceŽ
$model_1/lstm_lstm/lstm_cell/MatMul_4MatMul model_1/lstm_lstm/zeros:output:02model_1/lstm_lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2&
$model_1/lstm_lstm/lstm_cell/MatMul_4Ū
model_1/lstm_lstm/lstm_cell/addAddV2,model_1/lstm_lstm/lstm_cell/BiasAdd:output:0.model_1/lstm_lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2!
model_1/lstm_lstm/lstm_cell/add¬
#model_1/lstm_lstm/lstm_cell/SigmoidSigmoid#model_1/lstm_lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#model_1/lstm_lstm/lstm_cell/SigmoidŠ
,model_1/lstm_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp3model_1_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02.
,model_1/lstm_lstm/lstm_cell/ReadVariableOp_1·
1model_1/lstm_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       23
1model_1/lstm_lstm/lstm_cell/strided_slice_1/stack»
3model_1/lstm_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3model_1/lstm_lstm/lstm_cell/strided_slice_1/stack_1»
3model_1/lstm_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3model_1/lstm_lstm/lstm_cell/strided_slice_1/stack_2°
+model_1/lstm_lstm/lstm_cell/strided_slice_1StridedSlice4model_1/lstm_lstm/lstm_cell/ReadVariableOp_1:value:0:model_1/lstm_lstm/lstm_cell/strided_slice_1/stack:output:0<model_1/lstm_lstm/lstm_cell/strided_slice_1/stack_1:output:0<model_1/lstm_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2-
+model_1/lstm_lstm/lstm_cell/strided_slice_1ą
$model_1/lstm_lstm/lstm_cell/MatMul_5MatMul model_1/lstm_lstm/zeros:output:04model_1/lstm_lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2&
$model_1/lstm_lstm/lstm_cell/MatMul_5į
!model_1/lstm_lstm/lstm_cell/add_1AddV2.model_1/lstm_lstm/lstm_cell/BiasAdd_1:output:0.model_1/lstm_lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!model_1/lstm_lstm/lstm_cell/add_1²
%model_1/lstm_lstm/lstm_cell/Sigmoid_1Sigmoid%model_1/lstm_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%model_1/lstm_lstm/lstm_cell/Sigmoid_1Ź
model_1/lstm_lstm/lstm_cell/mulMul)model_1/lstm_lstm/lstm_cell/Sigmoid_1:y:0"model_1/lstm_lstm/zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2!
model_1/lstm_lstm/lstm_cell/mulŠ
,model_1/lstm_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp3model_1_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02.
,model_1/lstm_lstm/lstm_cell/ReadVariableOp_2·
1model_1/lstm_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1model_1/lstm_lstm/lstm_cell/strided_slice_2/stack»
3model_1/lstm_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   25
3model_1/lstm_lstm/lstm_cell/strided_slice_2/stack_1»
3model_1/lstm_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3model_1/lstm_lstm/lstm_cell/strided_slice_2/stack_2°
+model_1/lstm_lstm/lstm_cell/strided_slice_2StridedSlice4model_1/lstm_lstm/lstm_cell/ReadVariableOp_2:value:0:model_1/lstm_lstm/lstm_cell/strided_slice_2/stack:output:0<model_1/lstm_lstm/lstm_cell/strided_slice_2/stack_1:output:0<model_1/lstm_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2-
+model_1/lstm_lstm/lstm_cell/strided_slice_2ą
$model_1/lstm_lstm/lstm_cell/MatMul_6MatMul model_1/lstm_lstm/zeros:output:04model_1/lstm_lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2&
$model_1/lstm_lstm/lstm_cell/MatMul_6į
!model_1/lstm_lstm/lstm_cell/add_2AddV2.model_1/lstm_lstm/lstm_cell/BiasAdd_2:output:0.model_1/lstm_lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!model_1/lstm_lstm/lstm_cell/add_2„
 model_1/lstm_lstm/lstm_cell/ReluRelu%model_1/lstm_lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2"
 model_1/lstm_lstm/lstm_cell/ReluŲ
!model_1/lstm_lstm/lstm_cell/mul_1Mul'model_1/lstm_lstm/lstm_cell/Sigmoid:y:0.model_1/lstm_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!model_1/lstm_lstm/lstm_cell/mul_1Ķ
!model_1/lstm_lstm/lstm_cell/add_3AddV2#model_1/lstm_lstm/lstm_cell/mul:z:0%model_1/lstm_lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!model_1/lstm_lstm/lstm_cell/add_3Š
,model_1/lstm_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp3model_1_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02.
,model_1/lstm_lstm/lstm_cell/ReadVariableOp_3·
1model_1/lstm_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   23
1model_1/lstm_lstm/lstm_cell/strided_slice_3/stack»
3model_1/lstm_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3model_1/lstm_lstm/lstm_cell/strided_slice_3/stack_1»
3model_1/lstm_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3model_1/lstm_lstm/lstm_cell/strided_slice_3/stack_2°
+model_1/lstm_lstm/lstm_cell/strided_slice_3StridedSlice4model_1/lstm_lstm/lstm_cell/ReadVariableOp_3:value:0:model_1/lstm_lstm/lstm_cell/strided_slice_3/stack:output:0<model_1/lstm_lstm/lstm_cell/strided_slice_3/stack_1:output:0<model_1/lstm_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2-
+model_1/lstm_lstm/lstm_cell/strided_slice_3ą
$model_1/lstm_lstm/lstm_cell/MatMul_7MatMul model_1/lstm_lstm/zeros:output:04model_1/lstm_lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2&
$model_1/lstm_lstm/lstm_cell/MatMul_7į
!model_1/lstm_lstm/lstm_cell/add_4AddV2.model_1/lstm_lstm/lstm_cell/BiasAdd_3:output:0.model_1/lstm_lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!model_1/lstm_lstm/lstm_cell/add_4²
%model_1/lstm_lstm/lstm_cell/Sigmoid_2Sigmoid%model_1/lstm_lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%model_1/lstm_lstm/lstm_cell/Sigmoid_2©
"model_1/lstm_lstm/lstm_cell/Relu_1Relu%model_1/lstm_lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2$
"model_1/lstm_lstm/lstm_cell/Relu_1Ü
!model_1/lstm_lstm/lstm_cell/mul_2Mul)model_1/lstm_lstm/lstm_cell/Sigmoid_2:y:00model_1/lstm_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2#
!model_1/lstm_lstm/lstm_cell/mul_2³
/model_1/lstm_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   21
/model_1/lstm_lstm/TensorArrayV2_1/element_shape
!model_1/lstm_lstm/TensorArrayV2_1TensorListReserve8model_1/lstm_lstm/TensorArrayV2_1/element_shape:output:0*model_1/lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!model_1/lstm_lstm/TensorArrayV2_1r
model_1/lstm_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_1/lstm_lstm/time£
*model_1/lstm_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2,
*model_1/lstm_lstm/while/maximum_iterations
$model_1/lstm_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model_1/lstm_lstm/while/loop_counterē
model_1/lstm_lstm/whileWhile-model_1/lstm_lstm/while/loop_counter:output:03model_1/lstm_lstm/while/maximum_iterations:output:0model_1/lstm_lstm/time:output:0*model_1/lstm_lstm/TensorArrayV2_1:handle:0 model_1/lstm_lstm/zeros:output:0"model_1/lstm_lstm/zeros_1:output:0*model_1/lstm_lstm/strided_slice_1:output:0Imodel_1/lstm_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:09model_1_lstm_lstm_lstm_cell_split_readvariableop_resource;model_1_lstm_lstm_lstm_cell_split_1_readvariableop_resource3model_1_lstm_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"model_1_lstm_lstm_while_body_24875*.
cond&R$
"model_1_lstm_lstm_while_cond_24874*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
model_1/lstm_lstm/whileŁ
Bmodel_1/lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2D
Bmodel_1/lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape°
4model_1/lstm_lstm/TensorArrayV2Stack/TensorListStackTensorListStack model_1/lstm_lstm/while:output:3Kmodel_1/lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’*
element_dtype026
4model_1/lstm_lstm/TensorArrayV2Stack/TensorListStack„
'model_1/lstm_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2)
'model_1/lstm_lstm/strided_slice_3/stack 
)model_1/lstm_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)model_1/lstm_lstm/strided_slice_3/stack_1 
)model_1/lstm_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_1/lstm_lstm/strided_slice_3/stack_2
!model_1/lstm_lstm/strided_slice_3StridedSlice=model_1/lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:00model_1/lstm_lstm/strided_slice_3/stack:output:02model_1/lstm_lstm/strided_slice_3/stack_1:output:02model_1/lstm_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2#
!model_1/lstm_lstm/strided_slice_3
"model_1/lstm_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"model_1/lstm_lstm/transpose_1/permķ
model_1/lstm_lstm/transpose_1	Transpose=model_1/lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+model_1/lstm_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
model_1/lstm_lstm/transpose_1
model_1/lstm_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/lstm_lstm/runtime
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’@  2
model_1/flatten_1/Const¹
model_1/flatten_1/ReshapeReshape!model_1/lstm_lstm/transpose_1:y:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2
model_1/flatten_1/ReshapeĒ
(model_1/rnn_densef/MatMul/ReadVariableOpReadVariableOp1model_1_rnn_densef_matmul_readvariableop_resource*
_output_shapes
:	Ą*
dtype02*
(model_1/rnn_densef/MatMul/ReadVariableOpČ
model_1/rnn_densef/MatMulMatMul"model_1/flatten_1/Reshape:output:00model_1/rnn_densef/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
model_1/rnn_densef/MatMulÅ
)model_1/rnn_densef/BiasAdd/ReadVariableOpReadVariableOp2model_1_rnn_densef_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_1/rnn_densef/BiasAdd/ReadVariableOpĶ
model_1/rnn_densef/BiasAddBiasAdd#model_1/rnn_densef/MatMul:product:01model_1/rnn_densef/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
model_1/rnn_densef/BiasAdd
model_1/rnn_densef/SoftmaxSoftmax#model_1/rnn_densef/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
model_1/rnn_densef/Softmax
IdentityIdentity$model_1/rnn_densef/Softmax:softmax:0^model_1/lstm_lstm/while*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::22
model_1/lstm_lstm/whilemodel_1/lstm_lstm/while:T P
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ķ@

__inference__traced_save_28478
file_prefix0
,savev2_rnn_densef_kernel_read_readvariableop.
*savev2_rnn_densef_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop9
5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableopC
?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop7
3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_rnn_densef_kernel_m_read_readvariableop5
1savev2_adam_rnn_densef_bias_m_read_readvariableop@
<savev2_adam_lstm_lstm_lstm_cell_kernel_m_read_readvariableopJ
Fsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop>
:savev2_adam_lstm_lstm_lstm_cell_bias_m_read_readvariableop7
3savev2_adam_rnn_densef_kernel_v_read_readvariableop5
1savev2_adam_rnn_densef_bias_v_read_readvariableop@
<savev2_adam_lstm_lstm_lstm_cell_kernel_v_read_readvariableopJ
Fsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop>
:savev2_adam_lstm_lstm_lstm_cell_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d76e80b9c3ca424dbe5d91a87b7072be/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesø
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesö

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_rnn_densef_kernel_read_readvariableop*savev2_rnn_densef_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableop?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_rnn_densef_kernel_m_read_readvariableop1savev2_adam_rnn_densef_bias_m_read_readvariableop<savev2_adam_lstm_lstm_lstm_cell_kernel_m_read_readvariableopFsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop:savev2_adam_lstm_lstm_lstm_cell_bias_m_read_readvariableop3savev2_adam_rnn_densef_kernel_v_read_readvariableop1savev2_adam_rnn_densef_bias_v_read_readvariableop<savev2_adam_lstm_lstm_lstm_cell_kernel_v_read_readvariableopFsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop:savev2_adam_lstm_lstm_lstm_cell_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĻ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ć
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*¬
_input_shapes
: :	Ą:: : : : : :@:@:@: : : : :	Ą::@:@:@:	Ą::@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Ą: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$	 

_output_shapes

:@: 


_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Ą: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	Ą: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: 
ż

*__inference_rnn_densef_layer_call_fn_28131

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_rnn_densef_layer_call_and_return_conditional_losses_263302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’Ą::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’Ą
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
“
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_26311

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ī
É
)__inference_lstm_cell_layer_call_fn_28353

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_252182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ķ_
Ī
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_25777

inputs
lstm_cell_25679
lstm_cell_25681
lstm_cell_25683
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2č
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25679lstm_cell_25681lstm_cell_25683*
Tin

2*
Tout
2*M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_252182#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25679lstm_cell_25681lstm_cell_25683*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_25692*
condR
while_cond_25691*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeĘ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_25679*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addŚ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_25683*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ö
while_cond_27431
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_27431___redundant_placeholder0-
)while_cond_27431___redundant_placeholder1-
)while_cond_27431___redundant_placeholder2-
)while_cond_27431___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ž^
Ļ
while_body_26137
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ž^
Ļ
while_body_27940
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
`

lstm_lstm_while_body_26639 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_lstm_strided_slice_1_0[
Wtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_lstm_strided_slice_1Y
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeæ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yh
add_1AddV2lstm_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityi

Identity_1Identity"lstm_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"8
lstm_lstm_strided_slice_1lstm_lstm_strided_slice_1_0"°
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ž^
Ļ
while_body_27697
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 

ö
while_cond_26136
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_26136___redundant_placeholder0-
)while_cond_26136___redundant_placeholder1-
)while_cond_26136___redundant_placeholder2-
)while_cond_26136___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :’’’’’’’’’:’’’’’’’’’: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ū+

B__inference_model_1_layer_call_and_return_conditional_losses_26363
input_1
lstm_lstm_26298
lstm_lstm_26300
lstm_lstm_26302
rnn_densef_26341
rnn_densef_26343
identity¢!lstm_lstm/StatefulPartitionedCall¢"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_lstm_26298lstm_lstm_26300lstm_lstm_26302*
Tin
2*
Tout
2*+
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_260322#
!lstm_lstm/StatefulPartitionedCallÜ
flatten_1/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_263112
flatten_1/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0rnn_densef_26341rnn_densef_26343*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_rnn_densef_layer_call_and_return_conditional_losses_263302$
"rnn_densef/StatefulPartitionedCallĘ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26298*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addŚ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26302*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addČ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:T P
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ī
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27835

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimØ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimŖ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterŁ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_27697*
condR
while_cond_27696*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeŽ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addģ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ņ

)__inference_lstm_lstm_layer_call_fn_27592
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_257772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ėX

D__inference_lstm_cell_layer_call_and_return_conditional_losses_25132

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOp§
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicep
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1r
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:’’’’’’’’’2
mul|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2r
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_1]
add_3AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3r
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_2Ō
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addā
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ų+

B__inference_model_1_layer_call_and_return_conditional_losses_26432

inputs
lstm_lstm_26402
lstm_lstm_26404
lstm_lstm_26406
rnn_densef_26410
rnn_densef_26412
identity¢!lstm_lstm/StatefulPartitionedCall¢"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_26402lstm_lstm_26404lstm_lstm_26406*
Tin
2*
Tout
2*+
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_260322#
!lstm_lstm/StatefulPartitionedCallÜ
flatten_1/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_263112
flatten_1/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0rnn_densef_26410rnn_densef_26412*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_rnn_densef_layer_call_and_return_conditional_losses_263302$
"rnn_densef/StatefulPartitionedCallĘ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26402*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addŚ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26406*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addČ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ī
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_26032

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimØ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimŖ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterŁ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_25894*
condR
while_cond_25893*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeŽ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addģ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Č

)__inference_lstm_lstm_layer_call_fn_28089

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*+
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_260322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ū+

B__inference_model_1_layer_call_and_return_conditional_losses_26396
input_1
lstm_lstm_26366
lstm_lstm_26368
lstm_lstm_26370
rnn_densef_26374
rnn_densef_26376
identity¢!lstm_lstm/StatefulPartitionedCall¢"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_lstm_26366lstm_lstm_26368lstm_lstm_26370*
Tin
2*
Tout
2*+
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_262752#
!lstm_lstm/StatefulPartitionedCallÜ
flatten_1/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_263112
flatten_1/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0rnn_densef_26374rnn_densef_26376*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_rnn_densef_layer_call_and_return_conditional_losses_263302$
"rnn_densef/StatefulPartitionedCallĘ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26366*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addŚ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_26370*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addČ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:T P
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ī
É
)__inference_lstm_cell_layer_call_fn_28336

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_251322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ø
©
'__inference_model_1_layer_call_fn_27053

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_264322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ø
©
'__inference_model_1_layer_call_fn_27068

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:’’’’’’’’’*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_264802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
’
ė
while_body_25692
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_25716_0
lstm_cell_25718_0
lstm_cell_25720_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_25716
lstm_cell_25718
lstm_cell_25720¢!lstm_cell/StatefulPartitionedCall·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemü
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_25716_0lstm_cell_25718_0lstm_cell_25720_0*
Tin

2*
Tout
2*M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_252182#
!lstm_cell/StatefulPartitionedCallÖ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3¦

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4¦

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"$
lstm_cell_25716lstm_cell_25716_0"$
lstm_cell_25718lstm_cell_25718_0"$
lstm_cell_25720lstm_cell_25720_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ž^
Ļ
while_body_25894
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   23
1TensorArrayV2Read/TensorListGetItem/element_shapeµ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimŖ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split¦
lstm_cell/MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMulŖ
lstm_cell/MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1Ŗ
lstm_cell/MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2Ŗ
lstm_cell/MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim¬
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulplaceholder_2 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulplaceholder_2"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulplaceholder_2"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulplaceholder_2"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2æ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3k

Identity_4Identitylstm_cell/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :’’’’’’’’’:’’’’’’’’’: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 

ī
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_28078

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ī
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimØ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpĻ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimŖ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpĒ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd”
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_1”
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_2”
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2ø
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_1
lstm_cell/add_3AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shapeø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterŁ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_27940*
condR
while_cond_27939*K
output_shapes:
8: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeŽ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpĖ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constē
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Sum”
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ń82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xģ
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mul”
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xé
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addģ
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpé
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsĮ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sumµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulµ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ņ

)__inference_lstm_lstm_layer_call_fn_27581
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_256292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "ÆL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
?
input_14
serving_default_input_1:0’’’’’’’’’>

rnn_densef0
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:¾
Ž*
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
loss
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
W__call__
X_default_save_signature
*Y&call_and_return_all_conditional_losses"”(
_tf_keras_model({"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "name": "lstm_lstm", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["lstm_lstm", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "rnn_densef", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rnn_densef", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["rnn_densef", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 6]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "name": "lstm_lstm", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["lstm_lstm", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "rnn_densef", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rnn_densef", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["rnn_densef", 0, 0]]}}, "training_config": {"loss": ["categorical_crossentropy"], "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ń"ī
_tf_keras_input_layerĪ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ę
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"½
_tf_keras_rnn_layer{"class_name": "LSTM", "name": "lstm_lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 6]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 6]}}
Ć
	variables
trainable_variables
regularization_losses
	keras_api
\__call__
*]&call_and_return_all_conditional_losses"“
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"ļ
_tf_keras_layerÕ{"class_name": "Dense", "name": "rnn_densef", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "rnn_densef", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 320]}}
­
iter

beta_1

beta_2
	decay
 learning_ratemMmN!mO"mP#mQvRvS!vT"vU#vV"
	optimizer
 "
trackable_list_wrapper
C
!0
"1
#2
3
4"
trackable_list_wrapper
C
!0
"1
#2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
$layer_regularization_losses

%layers
	variables
&layer_metrics
'non_trainable_variables
trainable_variables
(metrics
	regularization_losses
W__call__
X_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
`serving_default"
signature_map
Ź	

!kernel
"recurrent_kernel
#bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
¹
-layer_regularization_losses

.layers

/states
0layer_metrics
	variables
1non_trainable_variables
trainable_variables
2metrics
regularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
3layer_regularization_losses

4layers
5layer_metrics
	variables
6non_trainable_variables
trainable_variables
7metrics
regularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
$:"	Ą2rnn_densef/kernel
:2rnn_densef/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
8layer_regularization_losses

9layers
:layer_metrics
	variables
;non_trainable_variables
trainable_variables
<metrics
regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*@2lstm_lstm/lstm_cell/kernel
6:4@2$lstm_lstm/lstm_cell/recurrent_kernel
&:$@2lstm_lstm/lstm_cell/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
­
?layer_regularization_losses

@layers
Alayer_metrics
)	variables
Bnon_trainable_variables
*trainable_variables
Cmetrics
+regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	Dtotal
	Ecount
F	variables
G	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
’
	Htotal
	Icount
J
_fn_kwargs
K	variables
L	keras_api"ø
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
D0
E1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
-
K	variables"
_generic_user_object
):'	Ą2Adam/rnn_densef/kernel/m
": 2Adam/rnn_densef/bias/m
1:/@2!Adam/lstm_lstm/lstm_cell/kernel/m
;:9@2+Adam/lstm_lstm/lstm_cell/recurrent_kernel/m
+:)@2Adam/lstm_lstm/lstm_cell/bias/m
):'	Ą2Adam/rnn_densef/kernel/v
": 2Adam/rnn_densef/bias/v
1:/@2!Adam/lstm_lstm/lstm_cell/kernel/v
;:9@2+Adam/lstm_lstm/lstm_cell/recurrent_kernel/v
+:)@2Adam/lstm_lstm/lstm_cell/bias/v
ź2ē
'__inference_model_1_layer_call_fn_26493
'__inference_model_1_layer_call_fn_26445
'__inference_model_1_layer_call_fn_27053
'__inference_model_1_layer_call_fn_27068Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ß
 __inference__wrapped_model_25006ŗ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ **¢'
%"
input_1’’’’’’’’’
Ö2Ó
B__inference_model_1_layer_call_and_return_conditional_losses_27038
B__inference_model_1_layer_call_and_return_conditional_losses_26786
B__inference_model_1_layer_call_and_return_conditional_losses_26363
B__inference_model_1_layer_call_and_return_conditional_losses_26396Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
)__inference_lstm_lstm_layer_call_fn_27592
)__inference_lstm_lstm_layer_call_fn_28089
)__inference_lstm_lstm_layer_call_fn_27581
)__inference_lstm_lstm_layer_call_fn_28100Õ
Ģ²Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ó2š
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27570
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27835
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_28078
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27327Õ
Ģ²Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ó2Š
)__inference_flatten_1_layer_call_fn_28111¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_flatten_1_layer_call_and_return_conditional_losses_28106¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_rnn_densef_layer_call_fn_28131¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_rnn_densef_layer_call_and_return_conditional_losses_28122¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2B0
#__inference_signature_wrapper_26534input_1
2
)__inference_lstm_cell_layer_call_fn_28353
)__inference_lstm_cell_layer_call_fn_28336¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Š2Ķ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_28233
D__inference_lstm_cell_layer_call_and_return_conditional_losses_28319¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
²2Æ
__inference_loss_fn_0_28366
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *¢ 
²2Æ
__inference_loss_fn_1_28379
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *¢ 
 __inference__wrapped_model_25006v!#"4¢1
*¢'
%"
input_1’’’’’’’’’
Ŗ "7Ŗ4
2

rnn_densef$!

rnn_densef’’’’’’’’’„
D__inference_flatten_1_layer_call_and_return_conditional_losses_28106]3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’Ą
 }
)__inference_flatten_1_layer_call_fn_28111P3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’Ą:
__inference_loss_fn_0_28366!¢

¢ 
Ŗ " :
__inference_loss_fn_1_28379"¢

¢ 
Ŗ " Ę
D__inference_lstm_cell_layer_call_and_return_conditional_losses_28233ż!#"¢}
v¢s
 
inputs’’’’’’’’’
K¢H
"
states/0’’’’’’’’’
"
states/1’’’’’’’’’
p
Ŗ "s¢p
i¢f

0/0’’’’’’’’’
EB

0/1/0’’’’’’’’’

0/1/1’’’’’’’’’
 Ę
D__inference_lstm_cell_layer_call_and_return_conditional_losses_28319ż!#"¢}
v¢s
 
inputs’’’’’’’’’
K¢H
"
states/0’’’’’’’’’
"
states/1’’’’’’’’’
p 
Ŗ "s¢p
i¢f

0/0’’’’’’’’’
EB

0/1/0’’’’’’’’’

0/1/1’’’’’’’’’
 
)__inference_lstm_cell_layer_call_fn_28336ķ!#"¢}
v¢s
 
inputs’’’’’’’’’
K¢H
"
states/0’’’’’’’’’
"
states/1’’’’’’’’’
p
Ŗ "c¢`

0’’’’’’’’’
A>

1/0’’’’’’’’’

1/1’’’’’’’’’
)__inference_lstm_cell_layer_call_fn_28353ķ!#"¢}
v¢s
 
inputs’’’’’’’’’
K¢H
"
states/0’’’’’’’’’
"
states/1’’’’’’’’’
p 
Ŗ "c¢`

0’’’’’’’’’
A>

1/0’’’’’’’’’

1/1’’’’’’’’’Ó
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27327!#"O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "2¢/
(%
0’’’’’’’’’’’’’’’’’’
 Ó
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27570!#"O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "2¢/
(%
0’’’’’’’’’’’’’’’’’’
 ¹
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_27835q!#"?¢<
5¢2
$!
inputs’’’’’’’’’

 
p

 
Ŗ ")¢&

0’’’’’’’’’
 ¹
D__inference_lstm_lstm_layer_call_and_return_conditional_losses_28078q!#"?¢<
5¢2
$!
inputs’’’’’’’’’

 
p 

 
Ŗ ")¢&

0’’’’’’’’’
 Ŗ
)__inference_lstm_lstm_layer_call_fn_27581}!#"O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "%"’’’’’’’’’’’’’’’’’’Ŗ
)__inference_lstm_lstm_layer_call_fn_27592}!#"O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "%"’’’’’’’’’’’’’’’’’’
)__inference_lstm_lstm_layer_call_fn_28089d!#"?¢<
5¢2
$!
inputs’’’’’’’’’

 
p

 
Ŗ "’’’’’’’’’
)__inference_lstm_lstm_layer_call_fn_28100d!#"?¢<
5¢2
$!
inputs’’’’’’’’’

 
p 

 
Ŗ "’’’’’’’’’²
B__inference_model_1_layer_call_and_return_conditional_losses_26363l!#"<¢9
2¢/
%"
input_1’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ²
B__inference_model_1_layer_call_and_return_conditional_losses_26396l!#"<¢9
2¢/
%"
input_1’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 ±
B__inference_model_1_layer_call_and_return_conditional_losses_26786k!#";¢8
1¢.
$!
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ±
B__inference_model_1_layer_call_and_return_conditional_losses_27038k!#";¢8
1¢.
$!
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
'__inference_model_1_layer_call_fn_26445_!#"<¢9
2¢/
%"
input_1’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
'__inference_model_1_layer_call_fn_26493_!#"<¢9
2¢/
%"
input_1’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
'__inference_model_1_layer_call_fn_27053^!#";¢8
1¢.
$!
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
'__inference_model_1_layer_call_fn_27068^!#";¢8
1¢.
$!
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’¦
E__inference_rnn_densef_layer_call_and_return_conditional_losses_28122]0¢-
&¢#
!
inputs’’’’’’’’’Ą
Ŗ "%¢"

0’’’’’’’’’
 ~
*__inference_rnn_densef_layer_call_fn_28131P0¢-
&¢#
!
inputs’’’’’’’’’Ą
Ŗ "’’’’’’’’’©
#__inference_signature_wrapper_26534!#"?¢<
¢ 
5Ŗ2
0
input_1%"
input_1’’’’’’’’’"7Ŗ4
2

rnn_densef$!

rnn_densef’’’’’’’’’
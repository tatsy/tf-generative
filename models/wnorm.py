import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.layers import base, utils
from tensorflow.python.ops import nn, standard_ops, array_ops, init_ops, nn_ops

class DenseWNorm(base.Layer):
    def __init__(self, units,
                 activation=None,
                 use_scale=True,
                 use_bias=True,
                 kernel_initializer=None,
                 scale_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 scale_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 scale_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(DenseWNorm, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)
        self.units = units
        self.activation = activation
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.scale_initializer = scale_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.scale_regularizer = scale_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.scale_constraint = scale_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)

        if self.use_scale:
            self.scale = self.add_variable('scale',
                                           shape=[self.units,],
                                           initializer=self.scale_initializer,
                                           regularizer=self.scale_regularizer,
                                           constraint=self.scale_constraint,
                                           dtype=self.dtype,
                                           trainable=True)
        else:
            self.scale = 1.0

        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[self.units,],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()

        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
                                                                   [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel)

        scaler = self.scale / tf.sqrt(tf.reduce_sum(tf.square(self.kernel), [0]))
        outputs = scaler * outputs

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)


def dense_wnorm(
        inputs, units,
        activation=None,
        use_scale=True,
        use_bias=True,
        kernel_initializer=None,
        scale_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        scale_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        scale_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):

    layer = DenseWNorm(units,
                       activation=activation,
                       use_scale=use_scale,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       scale_initializer=scale_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       scale_regularizer=scale_regularizer,
                       bias_regularizer=bias_regularizer,
                       activity_regularizer=activity_regularizer,
                       kernel_constraint=kernel_constraint,
                       scale_constraint=scale_constraint,
                       bias_constraint=bias_constraint,
                       trainable=trainable,
                       name=name,
                       dtype=inputs.dtype.base_dtype,
                       _scope=name,
                       _reuse=reuse)
    return layer.apply(inputs)


class _ConvWNorm(base.Layer):
    def __init__(self, rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_scale=True,
               use_bias=True,
               kernel_initializer=None,
               scale_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               scale_regularizer=None,
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               scale_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(_ConvWNorm, self).__init__(trainable=trainable, name=name,
                                         activity_regularizer=activity_regularizer,
                                         **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = utils.normalize_tuple(strides, rank, 'strides')
        self.padding = utils.normalize_padding(padding)
        self.data_format = utils.normalize_data_format(data_format)
        self.dilation_rate = utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activation
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.scale_initializer = scale_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.scale_regularizer = scale_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.scale_constraint = scale_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)

        if self.use_scale:
            self.scale = self.add_variable(name='scale',
                                           shape=(self.filters,),
                                           initializer=self.scale_initializer,
                                           regularizer=self.scale_regularizer,
                                           constraint=self.scale_constraint,
                                           trainable=True,
                                           dtype=self.dtype)
        else:
            self.scale = None

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None

        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})

        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.get_shape(),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format,
                                                  self.rank + 2))
        self.built = True

    def call(self, inputs):
        kernel_norm = nn.l2_normalize(self.kernel, [0, 1, 2])
        if self.use_scale:
            kernel_norm = tf.reshape(self.scale, [1, 1, 1, self.filters]) * kernel_norm
        outputs = self._convolution_op(inputs, kernel_norm)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    outputs_4d = array_ops.reshape(outputs,
                                                 [outputs_shape[0], outputs_shape[1],
                                                  outputs_shape[2] * outputs_shape[3],
                                                  outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
        else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] + new_space)

class Conv2DWNorm(_ConvWNorm):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_scale=True,
                 use_bias=True,
                 kernel_initializer=None,
                 scale_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 scale_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 scale_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv2DWNorm, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_scale=use_scale,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            scale_initializer=scale_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            scale_regularizer=scale_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            scale_constraint=scale_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs)


def conv2d_wnorm(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1),
           activation=None,
           use_scale=True,
           use_bias=True,
           kernel_initializer=None,
           scale_initializer=None,
           bias_initializer=init_ops.zeros_initializer(),
           kernel_regularizer=None,
           scale_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           scale_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           reuse=None):

    layer = Conv2DWNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_scale=use_scale,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        scale_initializer=scale_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        scale_regularizer=scale_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        scale_constraint=scale_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)

class Conv2DTransposeWNorm(Conv2DWNorm):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 activation=None,
                 use_scale=True,
                 use_bias=True,
                 kernel_initializer=None,
                 scale_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 scale_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 scale_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv2DTransposeWNorm, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_scale=use_scale,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            scale_initializer=scale_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            scale_regularizer=scale_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            scale_constraint=scale_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs)
        self.input_spec = base.InputSpec(ndim=4)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.input_spec = base.InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)

        if self.use_scale:
            self.scale = self.add_variable(name='scale',
                                           shape=(self.filters,),
                                           initializer=self.scale_initializer,
                                           regularizer=self.scale_regularizer,
                                           constraint=self.scale_constraint,
                                           trainable=True,
                                           dtype=self.dtype)
        else:
            self.scale = None


        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        out_height = utils.deconv_output_length(height,
                                                kernel_h,
                                                self.padding,
                                                stride_h)
        out_width = utils.deconv_output_length(width,
                                               kernel_w,
                                               self.padding,
                                               stride_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
            strides = (1, 1, stride_h, stride_w)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
            strides = (1, stride_h, stride_w, 1)

        output_shape_tensor = array_ops.stack(output_shape)

        kernel_norm = nn.l2_normalize(self.kernel, [0, 1, 3])
        if self.use_scale:
            kernel_norm = tf.reshape(self.scale, [1, 1, self.filters, 1]) * kernel_norm

        outputs = nn.conv2d_transpose(
            inputs,
            kernel_norm,
            output_shape_tensor,
            strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, ndim=4))

        if context.in_graph_mode():
            # Infer the static output shape:
            out_shape = inputs.get_shape().as_list()
            out_shape[c_axis] = self.filters
            out_shape[h_axis] = utils.deconv_output_length(out_shape[h_axis],
                                                           kernel_h,
                                                           self.padding,
                                                           stride_h)
            out_shape[w_axis] = utils.deconv_output_length(out_shape[w_axis],
                                                           kernel_w,
                                                           self.padding,
                                                           stride_w)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = utils.deconv_output_length(
            output_shape[h_axis], kernel_h, self.padding, stride_h)
        output_shape[w_axis] = utils.deconv_output_length(
            output_shape[w_axis], kernel_w, self.padding, stride_w)
        return tensor_shape.TensorShape(output_shape)


def conv2d_transpose_wnorm(
                     inputs,
                     filters,
                     kernel_size,
                     strides=(1, 1),
                     padding='valid',
                     data_format='channels_last',
                     activation=None,
                     use_scale=True,
                     use_bias=True,
                     kernel_initializer=None,
                     scale_initializer=None,
                     bias_initializer=init_ops.zeros_initializer(),
                     kernel_regularizer=None,
                     scale_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     scale_constraint=None,
                     bias_constraint=None,
                     trainable=True,
                     name=None,
                     reuse=None):
    layer = Conv2DTransposeWNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_scale=use_scale,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        scale_initializer=scale_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        scale_regularizer=scale_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        scale_constraint=scale_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)

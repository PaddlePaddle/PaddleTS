import paddle


class ConvLayer(paddle.nn.Layer):

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = paddle.nn.Conv1D(in_channels=c_in, out_channels=c_in,
            kernel_size=3, padding=2, padding_mode='circular')
        self.norm = paddle.nn.BatchNorm1D(num_features=c_in, momentum=1 - 
            0.1, epsilon=1e-05, weight_attr=None, bias_attr=None,
            use_global_stats=True)
        self.activation = paddle.nn.ELU()
        self.maxPool = paddle.nn.MaxPool1D(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.transpose(perm=[0, 2, 1]))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        x = x.transpose(perm=perm_0)
        return x


class EncoderLayer(paddle.nn.Layer):

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
        activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = paddle.nn.Conv1D(in_channels=d_model, out_channels=
            d_ff, kernel_size=1)
        self.conv2 = paddle.nn.Conv1D(in_channels=d_ff, out_channels=
            d_model, kernel_size=1)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.activation = (paddle.nn.functional.relu if activation ==
            'relu' else paddle.nn.functional.gelu)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau,
            delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(perm=[0,2,1]))))
        y = self.dropout(self.conv2(y).transpose(perm=[0,2,1]))
        return self.norm2(x + y), attn


class Encoder(paddle.nn.Layer):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = paddle.nn.LayerList(sublayers=attn_layers)
        self.conv_layers = paddle.nn.LayerList(sublayers=conv_layers
            ) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.
                attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta
                    =delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta
                    =delta)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(paddle.nn.Layer):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
        dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = paddle.nn.Conv1D(in_channels=d_model, out_channels=
            d_ff, kernel_size=1)
        self.conv2 = paddle.nn.Conv1D(in_channels=d_ff, out_channels=
            d_model, kernel_size=1)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.norm3 = paddle.nn.LayerNorm(normalized_shape=d_model, epsilon=
            1e-05, weight_attr=None, bias_attr=None)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.activation = (paddle.nn.functional.relu if activation ==
            'relu' else paddle.nn.functional.gelu)

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None,
        delta=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask,
            tau=tau, delta=None)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross,
            attn_mask=cross_mask, tau=tau, delta=delta)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(perm=[0,2,1]))))
        y = self.dropout(self.conv2(y).transpose(perm=[0,2,1]))
        return self.norm3(x + y)


class Decoder(paddle.nn.Layer):

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = paddle.nn.LayerList(sublayers=layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None,
        delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=
                tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x

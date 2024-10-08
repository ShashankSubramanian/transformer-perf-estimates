import numpy as np
import pandas as pd
from layers import *

def mlp_1d(b, l, e, f, parallelism={'m': 1}, topology={'t': 1}, system=None):
    """
    MLP layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim
                f: hidden dim
                element_size: in MB
                mask_element_size: in MB (for dropout)

    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)

    layer arithmetic:
        forward pass:
             X = XW + b
             (b,l,f) = (b,l,e) * (e,f) + (1,f)
             X = nonlinear(X)
             (b,l,f) = (b,l,f)
             X = dropout(X)
             (b,l,f) = (b,l,f) * (b,l,f) [random mask]
             X = linear(X)
             (b,l,e) = (b,l,f) * (f,e) + (1,e)
             X = dropout(X)
             (b,l,e) = (b,l,e) * (b,l,e) [random mask]

        backward pass:
             chain rule

    parallelism:
            X = XW + b
            (b,l,f/m) = (b,l,e) * (e,f/m) + (1,f/m)
            X = nonlinear(X)
            (b,l,f/m) = (b,l,f/m)
            X = dropout(X)
            (b,l,f/m) = (b,l,f/m) * (b,l,f/m) [random mask]
            X = linear(X)
            (b,l,e/m) = (b,l,f/m) * (f/m,e) + (1,e)
            X = dropout(X)
            (b,l/m,e) = (b,l/m,e) * (b,l/m,e) [random mask]
            X = norm(X)
            (b,l/m,e) = (b,l/m,e)

    comments:
    """

    summary = []
    m = parallelism['m']
    t = topology['t']

    ######################################################################################################################################################
    fc1 = Linear('fc1', b, l, e, f, parallelism={'dim1': 1, 'dim2': m}, topology={'t1': 1, 't2': t}, system=system)
    summary.append(fc1.get_stats())
    ######################################################################################################################################################
    fc1_bias = Bias('fc1-bias', b, l, f, parallelism={'dim1': 1, 'dim2': m}, topology={'t1': 1, 't2': t}, system=system)
    summary.append(fc1_bias.get_stats())
    ######################################################################################################################################################
    act1 = Act('act1', b * l * (f // m), system=system)
    summary.append(act1.get_stats())
    ######################################################################################################################################################
    dpr1 = DropOut('dpr1', b * l * (f // m), system=system)
    summary.append(dpr1.get_stats())
    ######################################################################################################################################################
    fc2 = Linear('fc2', b, l, f, e, parallelism={'dim1': m, 'dim2': 1}, topology={'t1': t, 't2': 1}, system=system)
    summary.append(fc2.get_stats())
    ######################################################################################################################################################
    fc2_bias = Bias('fc2-bias', b, l, e, parallelism={'dim1': 1, 'dim2': 1}, topology={'t1': 1, 't2': 1}, system=system)
    summary.append(fc2_bias.get_stats())
    ######################################################################################################################################################
    dpr2 = DropOut('dpr2', b * (l // m) * e, system=system)
    summary.append(dpr2.get_stats())
    ######################################################################################################################################################
    ln1 = LayerNorm('ln1', b, l, e, parallelism={'dim1': m}, topology={'t1': t}, system=system)
    summary.append(ln1.get_stats())
    ######################################################################################################################################################

    return pd.DataFrame(summary)

def mlp_2d(b, l, e, f, parallelism={'m1': 1, 'm2': 1}, topology={'t1': 1, 't2': 1}, system=None):
    """
    MLP layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim
                f: hidden dim
                element_size: in MB
                mask_element_size: in MB (for dropout)

    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)

    layer arithmetic:
        forward pass:
             X = XW + b
             (b,l,f) = (b,l,e) * (e,f) + (1,f)
             X = nonlinear(X)
             (b,l,f) = (b,l,f)
             X = dropout(X)
             (b,l,f) = (b,l,f) * (b,l,f) [random mask]
             X = linear(X)
             (b,l,e) = (b,l,f) * (f,e) + (1,e)
             X = dropout(X)
             (b,l,e) = (b,l,e) * (b,l,e) [random mask]

        backward pass:
             chain rule

    parallelism:
            X = XW + b
            (b,l/m2,f/m1) = (b,l/m2,e/m1) * (e/m2,f/m1) + (1,f/m1)
            X = nonlinear(X)
            (b,l/m2,f/m1) = (b,l/m2,f/m1)
            X = dropout(X)
            (b,l/m2,f/m1) = (b,l/m2,f/m1) * (b,l/m2,f/m1) [random mask]
            X = linear(X)
            (b,l/m2,e/m1) = (b,l/m2,f/m1) * (f/m2,e/m1) + (1,e/m1)
            X = dropout(X)
            (b,l/m2,e/m1) = (b,l/m2,e/m1) * (b,l/m2,e/m1) [random mask]
            X = norm(X)
            (b,l/m2,e/m1) = (b,l/m2,e/m1)

    comments:
    """

    summary = []
    m1 = parallelism['m1']
    t1 = topology['t1']
    m2 = parallelism['m2']
    t2 = topology['t2']

    ######################################################################################################################################################
    fc1 = LinearSumma('fc1', b, l, e, f, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(fc1.get_stats())
    ######################################################################################################################################################
    fc1_bias = Bias('fc1-bias', b, l, f, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(fc1_bias.get_stats())
    ######################################################################################################################################################
    act1 = Act('act1', b * (l // m2) * (f // m1), system=system)
    summary.append(act1.get_stats())
    ######################################################################################################################################################
    dpr1 = DropOut('dpr1', b * (l // m2) * (f // m1), system=system)
    summary.append(dpr1.get_stats())
    ######################################################################################################################################################
    fc2 = LinearSumma('fc2', b, l, f, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(fc2.get_stats())
    ######################################################################################################################################################
    fc2_bias = Bias('fc2-bias', b, l, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(fc2_bias.get_stats())
    ######################################################################################################################################################
    dpr2 = DropOut('dpr2', b * (l // m2) * (e // m1), system=system)
    summary.append(dpr2.get_stats())
    ######################################################################################################################################################
    ln1 = LayerNorm2D('ln1', b, l, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(ln1.get_stats())
    ######################################################################################################################################################

    return pd.DataFrame(summary)

def mlp_seqp(b, l, e, f, parallelism={'m1': 1, 'm2': 1}, topology={'t1': 1, 't2': 1}, system=None):
    """
    MLP layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim
                f: hidden dim
                element_size: in MB
                mask_element_size: in MB (for dropout)

    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)

    layer arithmetic:
        forward pass:
             X = XW + b
             (b,l,f) = (b,l,e) * (e,f) + (1,f)
             X = nonlinear(X)
             (b,l,f) = (b,l,f)
             X = dropout(X)
             (b,l,f) = (b,l,f) * (b,l,f) [random mask]
             X = linear(X)
             (b,l,e) = (b,l,f) * (f,e) + (1,e)
             X = dropout(X)
             (b,l,e) = (b,l,e) * (b,l,e) [random mask]

        backward pass:
             chain rule

    parallelism:
            X = XW + b
            (b,l/m2,f/m1) = (b,l/m2,e) * (e,f/m1) + (1,f/m1)
            X = nonlinear(X)
            (b,l/m2,f/m1) = (b,l/m2,f/m1)
            X = dropout(X)
            (b,l/m2,f/m1) = (b,l/m2,f/m1) * (b,l/m2,f/m1) [random mask]
            X = linear(X)
            (b,l/m2,e/m1) = (b,l/m2,f/m1) * (f/m1,e) + (1,e)
            X = dropout(X)
            (b,l/m2,e/m1) = (b,l/m2m1,e) * (b,l/m2m1,e) [random mask]
            X = norm(X)
            (b,l/m2m1,e) = (b,l/m2m1,e)

    comments:
    """

    summary = []
    m1 = parallelism['m1']
    t1 = topology['t1']
    m2 = parallelism['m2']
    t2 = topology['t2']

    ######################################################################################################################################################
    fc1 = LinearSeqp('fc1', b, l, e, f, parallelism={'dim1': 1, 'dim2': m1, 'dimseq': m2}, topology={'t1': 1, 't2': t1, 'tseq': t2}, system=system)
    summary.append(fc1.get_stats())
    ######################################################################################################################################################
    fc1_bias = Bias('fc1-bias', b, l, f, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(fc1_bias.get_stats())
    ######################################################################################################################################################
    act1 = Act('act1', b * (l // m2) * (f // m1), system=system)
    summary.append(act1.get_stats())
    ######################################################################################################################################################
    dpr1 = DropOut('dpr1', b * (l // m2) * (f // m1), system=system)
    summary.append(dpr1.get_stats())
    ######################################################################################################################################################
    fc2 = LinearSeqp('fc2', b, l, f, e, parallelism={'dim1': m1, 'dim2': 1, 'dimseq': m2}, topology={'t1': t1, 't2': 1, 'tseq': t2}, system=system)
    summary.append(fc2.get_stats())
    ######################################################################################################################################################
    fc2_bias = Bias('fc2-bias', b, (l // m2), e, parallelism={'dim1': m1, 'dim2': 1}, topology={'t1': t1, 't2': 1}, system=system)
    summary.append(fc2_bias.get_stats())
    ######################################################################################################################################################
    dpr2 = DropOut('dpr2', b * (l // (m1 * m2)) * e, system=system)
    summary.append(dpr2.get_stats())
    ######################################################################################################################################################
    ln1 = LayerNorm('ln1', b, (l // m2), e, parallelism={'dim1': m1}, topology={'t1': t1}, system=system)
    summary.append(ln1.get_stats())
    ######################################################################################################################################################

    return pd.DataFrame(summary)

def sa_1d(b, l, e, h, parallelism={'m': 1}, topology={'t': 1}, flash_attention=False, attention_recompute=False, system=None):
    """
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                h: number of attention heads
                element_size: in MB


    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)

    layer arithmetic:
        define: q = e/h
        forward pass:
             X = norm(X)
             Q = XW, K = XW, V = XW
             (b,l,h,q,3) = (b,l,e) * (e,3hq)
             A = QK'/sqrt(q)
             (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
             A = softmax(A)
             (b,h,l,l) = (b,h,l,l)
             A = dpr(A)
             Y = AV
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
             Y = VW
             (b,l,e) = (b,l,hq) * (hq,e)
             Y = dpr(Y)
             (b,l,e) = (b,l,e)
             Y = norm(Y)
             (b,l,e) = (b,l,e)

        backward pass:
             chain rule

        parallelism:
             Q = XW, K = XW, V = XW
             (b,l,h/m,q,3) = (b,l,e) * (e,3hq/m)
             A = QK'/sqrt(q)
             (b,h/m,l,l) = (b,h/m,l,q) * (b,h/m,q,l)
             A = softmax(A)
             (b,h/m,l,l) = (b,h/m,l,l)
             A = dpr(A)
             (b,h/m,l,l) = (b,h/m,l,l)
             Y = AV
             (b,h/m,l,q) = (b,h/m,l,l) * (b,h/m,l,q)
             Y = VW
             (b,l,e) = (b,l,hq/m) * (hq/m,e)
             Y = dpr(Y)
             (b,l/m,e) = (b,l/m,e)
             Y = norm(Y)
             (b,l/m,e) = (b,l/m,e)

        comments: use either flash attention or attention recompute
        for attention recompute, we recompute logits, softmax, dpr 
        to get the attention matrix in the attend layer
    """


    summary = []
    m = parallelism['m']
    t = topology['t']

    ######################################################################################################################################################
    qkv = Linear('qkv', b, l, e, (3 * e), parallelism={'dim1': 1, 'dim2': m}, topology={'t1': 1, 't2': t}, system=system)
    summary.append(qkv.get_stats())
    if flash_attention:
        ######################################################################################################################################################
        fusedla = FusedLA('fusedla', b, l, (e // h), h, parallelism={'dim1': m}, topology={'t1': t}, system=system)
        summary.append(fusedla.get_stats())
        ######################################################################################################################################################
    else:
        ######################################################################################################################################################
        # need to recompute logits if checkpointed, Q/K still stored
        logits = Logits('logits', b, l, (e // h), h, parallelism={'dim1': m}, topology={'t1': t}, recompute=attention_recompute, system=system)
        summary.append(logits.get_stats())
        ######################################################################################################################################################
        # need to recompute softmax if checkpointed, don't store input attention 
        softmax = Softmax('softmax', b, h, l, l, parallelism={'dim1': m}, topology={'t1': t}, recompute=attention_recompute,
                          remat=attention_recompute, system=system)
        summary.append(softmax.get_stats())
        ######################################################################################################################################################
        dpr_at = DropOut('dpr_at', b * (h // m) * l * l, recompute=attention_recompute, remat=attention_recompute,  system=system)
        summary.append(dpr_at.get_stats())
        ######################################################################################################################################################
        attend = Attend('attend', b, l, (e // h), h, parallelism={'dim1': m}, topology={'t1': t}, remat=attention_recompute,  system=system)
        summary.append(attend.get_stats())
        ######################################################################################################################################################
    vproj = Linear('vproj', b, l, e, e, parallelism={'dim1': m, 'dim2': 1}, topology={'t1': t, 't2': 1},  system=system)
    summary.append(vproj.get_stats())
    ######################################################################################################################################################
    vproj_bias = Bias('vproj-bias', b, l, e, parallelism={'dim1': 1, 'dim2': 1}, topology={'t1': 1, 't2': 1},  system=system)
    summary.append(vproj_bias.get_stats())
    ######################################################################################################################################################
    dpr_v = DropOut('dpr_v', b * (l // m) * e,  system=system)
    summary.append(dpr_v.get_stats())
    ######################################################################################################################################################
    ln2 = LayerNorm('ln2', b, l, e, parallelism={'dim1': m}, topology={'t1': t},  system=system)
    summary.append(ln2.get_stats())
    ######################################################################################################################################################

    return pd.DataFrame(summary)

def sa_2d(b, l, e, h, parallelism={'m1': 1, 'm2': 1}, topology={'t1': 1, 't2': 1}, system=None):
    """
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                h: number of attention heads
                element_size: in MB

    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)

    layer arithmetic:
        define: q = e/h
        forward pass:
             X = norm(X)
             Q = XW, K = XW, V = XW
             (b,l,h,q,3) = (b,l,e) * (e,3hq)
             A = QK'/sqrt(q)
             (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
             A = softmax(A)
             (b,h,l,l) = (b,h,l,l)
             A = dpr(A)
             Y = AV
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
             Y = VW
             (b,l,e) = (b,l,hq) * (hq,e)
             Y = dpr(Y)
             (b,l,e) = (b,l,e)
             Y = norm(Y)
             (b,l,e) = (b,l,e)

        backward pass:
             chain rule

        parallelism:
        Q = XW, K = XW, V = XW
        (b,h,l/m2,q/m1,3) = (b,l/m2,e/m1) * (e/m2,3hq/m1)
        A = QK'/sqrt(q)
        (b,h,l/m2,l/m1) = (b,h,l/m2,q/m1) * (b,h,q/m2,l/m1)
        A = softmax(A)
        (b,h,l/m2,l/m1) = (b,h,l/m2,l/m1)
        A = dpr(A)
        (b,h,l/m2,l/m1) = (b,h,l/m2,l/m1)
        Y = AV
        (b,h,l/m2,q/m1) = (b,h,l/m2,l/m1) * (b,h,l/m2,q/m1)
        Y = VW
        (b,l/m2,e/m1) = (b,l/m2,hq/m1) * (hq/m2,e/m1)
        Y = dpr(Y)
        (b,l/m2,e/m1) = (b,l/m2,e/m1)
        Y = norm(Y)
        (b,l/m2,e/m1) = (b,l/m2,e/m1)
    """


    summary = []
    m1 = parallelism['m1']
    t1 = topology['t1']
    m2 = parallelism['m2']
    t2 = topology['t2']

    ######################################################################################################################################################
    qkv = LinearSumma('qkv', b, l, e, (3 * e), parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1},  system=system)
    summary.append(qkv.get_stats())
    ######################################################################################################################################################
    logits = LogitsSumma('logits', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1},  system=system)
    summary.append(logits.get_stats())
    ######################################################################################################################################################
    softmax = Softmax2D('softmax', b, h, l, l, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1},  system=system)
    summary.append(softmax.get_stats())
    ######################################################################################################################################################
    dpr_at = DropOut('dpr_at', b * h * (l // m2) * (l // m1),  system=system)
    summary.append(dpr_at.get_stats())
    ######################################################################################################################################################
    attend = AttendSumma('attend', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1},  system=system)
    summary.append(attend.get_stats())
    ######################################################################################################################################################
    vproj = LinearSumma('vproj', b, l, e, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(vproj.get_stats())
    ######################################################################################################################################################
    vproj_bias = Bias('vproj-bias', b, l, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(vproj_bias.get_stats())
    ######################################################################################################################################################
    dpr_v = DropOut('dpr_v', b * (l // m2) * (e // m1), system=system)
    summary.append(dpr_v.get_stats())
    ######################################################################################################################################################
    ln2 = LayerNorm2D('ln2', b, l, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(ln2.get_stats())
    ######################################################################################################################################################

    return pd.DataFrame(summary)

def sa_2d_seqp(b, l, e, h, parallelism={'m1': 1, 'm2': 1}, topology={'t1': 1, 't2': 1}, flash_attention=True, system=None):
    """
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                h: number of attention heads
                element_size: in MB

    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)

    layer arithmetic:
        define: q = e/h
        forward pass:
             X = norm(X)
             Q = XW, K = XW, V = XW
             (b,l,h,q,3) = (b,l,e) * (e,3hq)
             A = QK'/sqrt(q)
             (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
             A = softmax(A)
             (b,h,l,l) = (b,h,l,l)
             A = dpr(A)
             Y = AV
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
             Y = VW
             (b,l,e) = (b,l,hq) * (hq,e)
             Y = dpr(Y)
             (b,l,e) = (b,l,e)
             Y = norm(Y)
             (b,l,e) = (b,l,e)

        backward pass:
             chain rule

        parallelism:
        Q = XW, K = XW, V = XW
        (b,h,l/m2,q/m1,3) = (b,l/m2,e/m1) * (e/m2,3hq/m1)
        A = QK'/sqrt(q)
        (b,h/m1,l/m2,l) = (b,h/m1,l/m2,q) * (b,h/m1,q,l/m2)
        A = softmax(A)
        (b,h/m1,l/m2,l) = (b,h/m1,l/m2,l)
        A = dpr(A)
        (b,h/m1,l/m2,l) = (b,h/m1,l/m2,l)
        Y = AV
        (b,h/m1,l/m2,q) = (b,h/m1,l/m2,q) * (b,h/m1,l/m2,q)
        Y = VW
        (b,l/m2,e/m1) = (b,l/m2,hq/m1) * (hq/m2,e/m1)
        Y = dpr(Y)
        (b,l/m2,e/m1) = (b,l/m2,e/m1)
        Y = norm(Y)
        (b,l/m2,e/m1) = (b,l/m2,e/m1)
    """


    summary = []
    m1 = parallelism['m1']
    t1 = topology['t1']
    m2 = parallelism['m2']
    t2 = topology['t2']

    ######################################################################################################################################################
    qkv = LinearSumma('qkv', b, l, e, (3 * e), parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(qkv.get_stats())

    if flash_attention:
        ######################################################################################################################################################
        fusedla = FusedLASeqp('fusedla', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
        summary.append(fusedla.get_stats())
        ######################################################################################################################################################
    else:
        ######################################################################################################################################################
        logits = LogitsSeqp('logits', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
        summary.append(logits.get_stats())
        ######################################################################################################################################################
        softmax = Softmax('softmax', b, h, (l // m2), l, parallelism={'dim1': m1}, topology={'t1': t1}, system=system)
        summary.append(softmax.get_stats())
        ######################################################################################################################################################
        dpr_at = DropOut('dpr_at', b * (h // m1) * (l // m2) * l, system=system)
        summary.append(dpr_at.get_stats())
        ######################################################################################################################################################
        attend = AttendSeqp('attend', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
        summary.append(attend.get_stats())
        ######################################################################################################################################################
    vproj = LinearSumma('vproj', b, l, e, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(vproj.get_stats())
    ######################################################################################################################################################
    vproj_bias = Bias('vproj-bias', b, l, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(vproj_bias.get_stats())
    ######################################################################################################################################################
    dpr_v = DropOut('dpr_v', b * (l // m2) * (e // m1), system=system)
    summary.append(dpr_v.get_stats())
    ######################################################################################################################################################
    ln2 = LayerNorm2D('ln2', b, l, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(ln2.get_stats())
    ######################################################################################################################################################

    return pd.DataFrame(summary)

def sa_seqp(b, l, e, h, parallelism={'m1': 1, 'm2': 1}, topology={'t1': 1, 't2': 1}, flash_attention=True, system=None):
    """
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                h: number of attention heads
                element_size: in MB

    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)

    layer arithmetic:
        define: q = e/h
        forward pass:
             X = norm(X)
             Q = XW, K = XW, V = XW
             (b,l,h,q,3) = (b,l,e) * (e,3hq)
             A = QK'/sqrt(q)
             (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
             A = softmax(A)
             (b,h,l,l) = (b,h,l,l)
             A = dpr(A)
             Y = AV
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
             Y = VW
             (b,l,e) = (b,l,hq) * (hq,e)
             Y = dpr(Y)
             (b,l,e) = (b,l,e)
             Y = norm(Y)
             (b,l,e) = (b,l,e)

        backward pass:
             chain rule

        parallelism:
        Q = XW, K = XW, V = XW
        (b,h,l/m2,q/m1,3) = (b,l/m2,e) * (e,3hq/m1)
        A = QK'/sqrt(q)
        (b,h/m1,l/m2,l) = (b,h/m1,l/m2,q) * (b,h/m1,q,l/m2)
        A = softmax(A)
        (b,h/m1,l/m2,l) = (b,h/m1,l/m2,l)
        A = dpr(A)
        (b,h/m1,l/m2,l) = (b,h/m1,l/m2,l)
        Y = AV
        (b,h/m1,l/m2,q) = (b,h/m1,l/m2,q) * (b,h/m1,l/m2,q)
        Y = VW
        (b,l/m2m1,e) = (b,l/m2,hq/m1) * (hq/m1,e)
        Y = dpr(Y)
        (b,l/m2m1,e) = (b,l/m2m1,e)
        Y = norm(Y)
        (b,l/m2m1,e) = (b,l/m2m1,e)
    """


    summary = []
    m1 = parallelism['m1']
    t1 = topology['t1']
    m2 = parallelism['m2']
    t2 = topology['t2']

    ######################################################################################################################################################
    qkv = LinearSeqp('qkv', b, l, e, (3 * e), parallelism={'dim1': 1, 'dim2': m1, 'dimseq': m2}, topology={'t1': 1, 't2': t1, 'tseq': t2}, system=system)
    summary.append(qkv.get_stats())
    ######################################################################################################################################################
    if flash_attention:
        ######################################################################################################################################################
        fusedla = FusedLASeqp('fusedla', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
        summary.append(fusedla.get_stats())
        ######################################################################################################################################################
    else:
        ######################################################################################################################################################
        logits = LogitsSeqp('logits', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
        summary.append(logits.get_stats())
        ######################################################################################################################################################
        softmax = Softmax('softmax', b, h, (l // m2), l, parallelism={'dim1': m1}, topology={'t1': t1}, system=system)
        summary.append(softmax.get_stats())
        ######################################################################################################################################################
        dpr_at = DropOut('dpr_at', b * (h // m1) * (l // m2) * l, system=system)
        summary.append(dpr_at.get_stats())
        ######################################################################################################################################################
        attend = AttendSeqp('attend', b, l, (e // h), h, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
        summary.append(attend.get_stats())
        ######################################################################################################################################################
    vproj = LinearSeqp('vproj', b, l, e, e, parallelism={'dim1': m1, 'dim2': 1, 'dimseq': m2}, topology={'t1': t1, 't2': 1, 'tseq': t2}, system=system)
    summary.append(vproj.get_stats())
    ######################################################################################################################################################
    vproj_bias = Bias('vproj-bias', b, l, e, parallelism={'dim1': m2, 'dim2': m1}, topology={'t1': t2, 't2': t1}, system=system)
    summary.append(vproj_bias.get_stats())
    ######################################################################################################################################################
    dpr_v = DropOut('dpr_v', b * (l // (m1 * m2)) * e, system=system)
    summary.append(dpr_v.get_stats())
    ######################################################################################################################################################
    ln2 = LayerNorm('ln2', b, (l // m2), e, parallelism={'dim1': m1}, topology={'t1': t1}, system=system)
    summary.append(ln2.get_stats())
    ######################################################################################################################################################

    return pd.DataFrame(summary)

def dataparallel(modules=[],
       depth=1,
       dp=1,
       t_dp=1,
       overlap=True,
       system=None):
    summary = []
    dp_comms = DataParallel('dp_comms', modules=modules, depth=depth, dp=dp, t_dp=t_dp, overlap=overlap, system=system)
    summary.append(dp_comms.get_stats())
    return pd.DataFrame(summary)

def pipelineparallel(modules=[],
       number_micro_batches=1,
       comm_vol=0,
       pp=1,
       t_pp=1,
       overlap=False,
       system=None):
    summary = []
    pp_comms = PipelineParallel('pp_comms', modules=modules, number_micro_batches=number_micro_batches, comm_vol=comm_vol, pp=pp, t_pp=t_pp, overlap=overlap, system=system)
    summary.append(pp_comms.get_stats())
    return pd.DataFrame(summary)

def print_df(df_mlp, df_sa):
    ''' prints stats of transformer dataframe '''
    cols = df_mlp.columns.tolist()
    layer_track_cols = ['activation_buffer', 'weights_mem',
                        'weights_grad_mem', 'flops_fwd', 'flops_bwd',
                        't_fwd', 't_fwd_comm', 't_bwd', 't_bwd_comm']
    display(df_mlp[cols])
    display(df_sa[cols])

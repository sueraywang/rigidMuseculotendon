import warp as wp
WORLD_FIX_CONSTRAINT = wp.constant(0)
WORLD_CONTACT_CONSTRAINT = wp.constant(1)
WORLD_ROTATE_CONSTRAINT = wp.constant(2)
WORLD_ROTATE_TARGET_CONSTRAINT = wp.constant(3)
BODY_FIX_CONSTRAINT = wp.constant(4)
BODY_CONTACT_CONSTRAINT = wp.constant(5)
BODY_ROTATE_CONSTRAINT = wp.constant(6)
BODY_ROTATE_TARGET_CONSTRAINT = wp.constant(7)
BODY_MUSCLE_CONSTRAINT = wp.constant(8)

import numpy as np
from DataTypes import *
from CollisionDetection import collisionDetectionGroundCuboid, collisionDetectionCuboidCuboid
from utils import SIM_NUM, CONTACT_MAX, EPS_SMALL, VIA_POINT_MAX, getContactFrame, gamma
ROTATE_OFFSET = 1E-2

@wp.kernel
def initConstraintsContact(
    con_count: wp.array(dtype=INT_DATA_TYPE),
    con_type: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body1_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body2_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xl1: wp.array(dtype=Vec3, ndim=2),
    xl2: wp.array(dtype=Vec3, ndim=2),
    normal: wp.array(dtype=Vec3, ndim=2),
    # outputs
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
):
    i,j = wp.tid()
    if i > con_count[j]:
        return
    if(con_type[i][j] == WORLD_CONTACT_CONSTRAINT):
        b_i = body2_ind[i][j]
        cf = getContactFrame(normal[i][j])
        x_w = wp.transform_vector(q[b_i][j], xl2[i][j])
        J2[i][j] = cf * gamma(x_w)

        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
        M_p = I[b_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i][j], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)

        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel
    elif(con_type[i][j] == BODY_CONTACT_CONSTRAINT):
        b1_i = body1_ind[i][j]
        b2_i = body2_ind[i][j]
        cf = getContactFrame(normal[i][j])
        J1[i][j] = cf * gamma(wp.transform_vector(q[b1_i][j], xl1[i][j]))
        J2[i][j] = cf * gamma(wp.transform_vector(q[b2_i][j], xl2[i][j]))
        
        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b1_i][0],I[b1_i][1],I[b1_i][2])
        M_p = I[b1_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl1[i][j], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b1_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w1[i][j] = w_kernel
        J_div_m1[i][j] = J_div_m_kernel
        
        M_r = Vec3(I[b2_i][0],I[b2_i][1],I[b2_i][2])
        M_p = I[b2_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i][j], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b2_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel

@wp.kernel
def solveConstraintsContact(
    con_count: wp.array(dtype=INT_DATA_TYPE),
    con_type: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body1_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body2_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
    d: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    h: FP_DATA_TYPE,
    mu: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    #outputs
    lambdas: wp.array(dtype=Vec3, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2),
):
    j = wp.tid()
    for i in range(con_count[j]):
        if(con_type[i][j] == WORLD_CONTACT_CONSTRAINT):
            b_i = body2_ind[i][j]
            d_ij = d[i][j]
            phi_ij = phi[b_i][j]
            w_ij = w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias = phi_dt[b_i][j] / h
            J_ij = J2[i][j]
            J_div_m_ij = J_div_m2[i][j]

            c = -wp.dot(J_ij[0], (phi_ij +bias)) + d_ij / h
            # wp.printf("c: %f = %f + %f\n", c, wp.dot(J_ij[0], (phi_ij +bias)), d_ij / h)
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            # wp.printf("phi: %f %f %f %f %f %f\n", phi_ij[0], phi_ij[1], phi_ij[2], phi_ij[3], phi_ij[4], phi_ij[5])
            # wp.printf("Jijk: %f %f %f %f %f %f\n", J_ij[k][0], J_ij[k][1], J_ij[k][2], J_ij[k][3], J_ij[k][4], J_ij[k][5])
            dlambda_nor = -c / w_ij[0]
            # wp.printf("dlambda: %f \n", dlambda)
            if(lambdas_ij[0] + dlambda_nor < 0.0):
                dlambda_nor = -lambdas_ij[0]
            lambdas_ij[0] += dlambda_nor
            phi_ij -= J_div_m_ij[0] * dlambda_nor
            
            c1 = -wp.dot(J_ij[1], (phi_ij +bias)) 
            c2 = -wp.dot(J_ij[2], (phi_ij +bias)) 
            # wp.printf("c: %f = %f + %f\n", c, wp.dot(J_ij[k], (phi_ij +bias)), d_ij[k] / h)
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            # wp.printf("phi: %f %f %f %f %f %f\n", phi_ij[0], phi_ij[1], phi_ij[2], phi_ij[3], phi_ij[4], phi_ij[5])
            # wp.printf("Jijk: %f %f %f %f %f %f\n", J_ij[k][0], J_ij[k][1], J_ij[k][2], J_ij[k][3], J_ij[k][4], J_ij[k][5])
            dlambda_tan1 = -c1 / w_ij[1]
            dlambda_tan2 = -c2 / w_ij[2]
            # wp.printf("dlambda: %f \n", dlambda)
            lambda_tan = Vec2(lambdas_ij[1]+ dlambda_tan1, lambdas_ij[2]+ dlambda_tan2)
            lambda_tan_norm = wp.math.norm_l2(lambda_tan)
            if(lambda_tan_norm > wp.max(EPS_SMALL, mu[i][j] * lambdas_ij[0])):
                dlambda_tan1 = mu[i][j] * lambdas_ij[0] * lambda_tan[0] / lambda_tan_norm - lambdas_ij[1]
                dlambda_tan2 = mu[i][j] * lambdas_ij[0] * lambda_tan[1] / lambda_tan_norm - lambdas_ij[2]
            lambdas_ij[1] += dlambda_tan1
            lambdas_ij[2] += dlambda_tan2
            phi_ij -= J_div_m_ij[1] * dlambda_tan1 + J_div_m_ij[2] * dlambda_tan2
            lambdas[i][j] = lambdas_ij
            phi[b_i][j] = phi_ij
        elif(con_type[i][j] == BODY_CONTACT_CONSTRAINT):
            b1_i = body1_ind[i][j]
            b2_i = body2_ind[i][j]
            d_ij = d[i][j]
            phi1_ij = phi[b1_i][j]
            phi2_ij = phi[b2_i][j]
            w_ij = w1[i][j] + w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias1 = phi_dt[b1_i][j] / h
            bias2 = phi_dt[b2_i][j] / h
            J1_ij = J1[i][j]
            J_div_m1_ij = J_div_m1[i][j]
            J2_ij = J2[i][j]
            J_div_m2_ij = J_div_m2[i][j]

            c = wp.dot(J1_ij[0], (phi1_ij +bias1)) - wp.dot(J2_ij[0], (phi2_ij +bias2)) + d_ij / h
            # wp.printf("c: %f = %f - %f + %f \n", c, wp.dot(J1_ij[0], (phi1_ij +bias1)), wp.dot(J2_ij[0], (phi2_ij +bias2)), d_ij / h)
            # wp.printf("w_ij: %f \n", w_ij[k])
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias2: %f %f %f %f %f %f\n", bias2[0], bias2[1], bias2[2], bias2[3], bias2[4], bias2[5])
            # wp.printf("phi1: %f %f %f %f %f %f\n", phi1_ij[0], phi1_ij[1], phi1_ij[2], phi1_ij[3], phi1_ij[4], phi1_ij[5])
            # wp.printf("phi2: %f %f %f %f %f %f\n", phi2_ij[0], phi2_ij[1], phi2_ij[2], phi2_ij[3], phi2_ij[4], phi2_ij[5])
            # wp.printf("J1ijk: %f %f %f %f %f %f\n", J1_ij[k][0], J1_ij[k][1], J1_ij[k][2], J1_ij[k][3], J1_ij[k][4], J1_ij[k][5])
            # wp.printf("J2ij0: %f %f %f %f %f %f\n", J2_ij[0][0], J2_ij[0][1], J2_ij[0][2], J2_ij[0][3], J2_ij[0][4], J2_ij[0][5])
            dlambda_nor = -c / w_ij[0]
            # wp.printf("dlambda: %f \n", dlambda)
            if(lambdas_ij[0] + dlambda_nor < 0.0):
                dlambda_nor = -lambdas_ij[0]
            lambdas_ij[0] += dlambda_nor
            phi1_ij += J_div_m1_ij[0] * dlambda_nor
            phi2_ij -= J_div_m2_ij[0] * dlambda_nor
            
            c1 = wp.dot(J1_ij[1], (phi1_ij +bias1)) - wp.dot(J2_ij[1], (phi2_ij +bias2))
            c2 = wp.dot(J1_ij[2], (phi1_ij +bias1)) - wp.dot(J2_ij[2], (phi2_ij +bias2))
            # wp.printf("c: %f = %f - %f\n", c, wp.dot(J1_ij[k], (phi1_ij +bias1)), wp.dot(J2_ij[k], (phi2_ij +bias2)))
            # wp.printf("w_ij: %f \n", w_ij[k])
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            # wp.printf("phi1: %f %f %f %f %f %f\n", phi1_ij[0], phi1_ij[1], phi1_ij[2], phi1_ij[3], phi1_ij[4], phi1_ij[5])
            # wp.printf("phi2: %f %f %f %f %f %f\n", phi2_ij[0], phi2_ij[1], phi2_ij[2], phi2_ij[3], phi2_ij[4], phi2_ij[5])
            # wp.printf("J1ijk: %f %f %f %f %f %f\n", J1_ij[k][0], J1_ij[k][1], J1_ij[k][2], J1_ij[k][3], J1_ij[k][4], J1_ij[k][5])
            # wp.printf("J2ijk: %f %f %f %f %f %f\n", J2_ij[k][0], J2_ij[k][1], J2_ij[k][2], J2_ij[k][3], J2_ij[k][4], J2_ij[k][5])
            dlambda_tan1 = -c1 / w_ij[1]
            dlambda_tan2 = -c2 / w_ij[2]
            # wp.printf("dlambda: %f \n", dlambda)
            lambda_tan = Vec2(lambdas_ij[1]+ dlambda_tan1, lambdas_ij[2]+ dlambda_tan2)
            lambda_tan_norm = wp.math.norm_l2(lambda_tan)
            if(lambda_tan_norm > wp.max(EPS_SMALL, mu[i][j] * lambdas_ij[0])):
                dlambda_tan1 = mu[i][j] * lambdas_ij[0] * lambda_tan[0] / lambda_tan_norm - lambdas_ij[1]
                dlambda_tan2 = mu[i][j] * lambdas_ij[0] * lambda_tan[1] / lambda_tan_norm - lambdas_ij[2]
            lambdas_ij[1] += dlambda_tan1
            lambdas_ij[2] += dlambda_tan2
            phi1_ij += J_div_m1_ij[1] * dlambda_tan1 + J_div_m1_ij[2] * dlambda_tan2
            phi2_ij -= J_div_m2_ij[1] * dlambda_tan1 + J_div_m2_ij[2] * dlambda_tan2
            lambdas[i][j] = lambdas_ij
            phi[b1_i][j] = phi1_ij
            phi[b2_i][j] = phi2_ij
            
@wp.kernel
def initConstraintsJoint(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body1_ind: wp.array(dtype=INT_DATA_TYPE),
    body2_ind: wp.array(dtype=INT_DATA_TYPE),
    phi: wp.array(dtype=Vec6, ndim=2),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xl1: wp.array(dtype=Vec3),
    xl2: wp.array(dtype=Vec3),
    axis1: wp.array(dtype=Vec3),
    axis2: wp.array(dtype=Vec3),
    b1: wp.array(dtype=Vec3),
    b2: wp.array(dtype=Vec3),
    theta_target: wp.array(dtype=FP_DATA_TYPE),
    oemga_target: wp.array(dtype=FP_DATA_TYPE),
    kp: wp.array(dtype=FP_DATA_TYPE),
    kd: wp.array(dtype=FP_DATA_TYPE),
    limits: wp.array(dtype=Vec2),
    limit_signs: wp.array(dtype=FP_DATA_TYPE),
    h: FP_DATA_TYPE,
    # outputs
    d: wp.array(dtype=Vec3, ndim=2),
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
):
    i,j = wp.tid()
    if i > con_count:
        return
    
    if(con_type[i] == WORLD_FIX_CONSTRAINT):
        b_i = body2_ind[i]
        cf = getContactFrame(axis1[i])
        x_w = wp.transform_vector(q[b_i][j], xl2[i])
        # q_ij = q[b_i][j]
        # wp.printf("q: %f %f %f %f %f %f %f\n", q_ij[0], q_ij[1], q_ij[2], q_ij[3], q_ij[4], q_ij[5], q_ij[6])
        # wp.printf("axis: %f %f %f\n", axis[i][0], axis[i][1], axis[i][2])
        # wp.printf("x_l: %f %f %f\n", xl[i][0], xl[i][1], xl[i][2])
        # wp.printf("x_w: %f %f %f\n", x_w[0], x_w[1], x_w[2])

        d[i][j] = cf * (xl1[i] - x_w - q[b_i][j].p)
        J2[i][j] = cf * gamma(x_w)
        # J2_ij = J2[i][j]
        # wp.printf("J2:\n %f %f %f %f %f %f\n", J2_ij[0][0], J2_ij[0][1], J2_ij[0][2], J2_ij[0][3], J2_ij[0][4], J2_ij[0][5])
        # wp.printf("%f %f %f %f %f %f\n", J2_ij[1][0], J2_ij[1][1], J2_ij[1][2], J2_ij[1][3], J2_ij[1][4], J2_ij[1][5])
        # wp.printf("%f %f %f %f %f %f\n", J2_ij[2][0], J2_ij[2][1], J2_ij[2][2], J2_ij[2][3], J2_ij[2][4], J2_ij[2][5])

        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
        M_p = I[b_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)

        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel
    elif(con_type[i] == BODY_FIX_CONSTRAINT):
        b1_i = body1_ind[i]
        b2_i = body2_ind[i]
        cf = getContactFrame(wp.transform_vector(q[b1_i][j], axis1[i]))
        d[i][j] = cf * (wp.transform_point(q[b1_i][j], xl1[i]) - wp.transform_point(q[b2_i][j], xl2[i]))
        J1[i][j] = cf * gamma(wp.transform_vector(q[b1_i][j], xl1[i]))
        J2[i][j] = cf * gamma(wp.transform_vector(q[b2_i][j], xl2[i]))
        
        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b1_i][0],I[b1_i][1],I[b1_i][2])
        M_p = I[b1_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl1[i], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b1_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w1[i][j] = w_kernel
        J_div_m1[i][j] = J_div_m_kernel
        
        M_r = Vec3(I[b2_i][0],I[b2_i][1],I[b2_i][2])
        M_p = I[b2_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b2_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel
    elif(con_type[i] == WORLD_ROTATE_CONSTRAINT or con_type[i] == WORLD_ROTATE_TARGET_CONSTRAINT):
            b_i = body2_ind[i]
            cf = getContactFrame(axis1[i])

            w_kernel = Vec3()
            J_div_m_kernel = Mat36()
            J_kernel = Mat36()
            M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
            d_ij = Vec3(0.0)
            for k in range(3):
                n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
                w_kernel[k] = wp.dot(n_l, wp.cw_div(n_l, M_r))
                rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(n_l, M_r))
                J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], 0.0, 0.0, 0.0)
                J_kernel[k] = Vec6(cf[k][0], cf[k][1], cf[k][2], 0.0, 0.0, 0.0)

            w2[i][j] = w_kernel
            J2[i][j] = J_kernel
            J_div_m2[i][j] = J_div_m_kernel

            # r_axis, dtheta = wp.quat_to_axis_angle(wp.quat_inverse(q[b_i][j].q))
            # # wp.printf("dq: %f %f %f %f\n", q[b_i][j].q[0], q[b_i][j].q[1], q[b_i][j].q[2], q[b_i][j].q[3])
            # d[i][j] = cf * r_axis * dtheta
            
            d_ij = cf * wp.cross(wp.transform_vector(q[b_i][j], axis2[i]), axis1[i])
            n1 = b1[i]
            n2 = wp.quat_rotate(q[b_i][j].q, b2[i])
            d_target = wp.asin(wp.dot(cf[0], wp.cross(n2, n1)))
            if(wp.dot(n1, n2) < 0.0):
                d_target = wp.PI - d_target
            d_target = wp.atan2(wp.sin(d_target), wp.cos(d_target))  # map to [-pi, pi]
            if(con_type[i] == WORLD_ROTATE_CONSTRAINT):
                if(d_target > limits[i][0] - ROTATE_OFFSET):
                    d_ij[0] = d_target - limits[i][0]
                    limit_signs[i] = -1.0
                elif(d_target < limits[i][1] + ROTATE_OFFSET):
                    d_ij[0] = d_target - limits[i][1]
                    limit_signs[i] = 1.0
                else:
                    d_ij[0] = 0.0
                    limit_signs[i] = 0.0
            elif(con_type[i] == WORLD_ROTATE_TARGET_CONSTRAINT):
                # dtheta_align = wp.cross(wp.transform_vector(q[b_i][j], axis2[i]), axis1[i])
                # dtheta_align_norm = wp.norm_l2(dtheta_align)
                # if dtheta_align_norm > EPS_SMALL:
                #     dq_align = wp.quat_from_axis_angle(dtheta_align / dtheta_align_norm, dtheta_align_norm)
                # else:
                #     dq_align = wp.quat_identity()
                # d_target = wp.dot(cf[0], wp.cross(wp.quat_rotate(dq_align * q[b_i][j].q, b2[i]), b1[i]))
                d_omega = -wp.dot(J_kernel[0], phi[b_i][j])
                a = h/(h*(h*kp[i]+kd[i])*wp.dot(J_kernel[0], -J_div_m_kernel[0]) + 1.0)
                d_ij[0] = a * h * (kp[i]*((d_target - theta_target[i]) + d_omega * h) + kd[i]*(d_omega - oemga_target[i])) - d_omega * h
                # wp.printf("d_target: %f\n", d_target)
                # wp.printf("d_omega: %f\n", d_omega) 
                # wp.printf("a: %f\n", a) 
                # wp.printf("dq_align: %f %f %f %f\n", dq_align[0], dq_align[1], dq_align[2], dq_align[3]) 
                # wp.printf("kp: %f\n", kp[i]) 
                # wp.printf("kd: %f\n", kd[i]) 
                # wp.printf("h*kp[i]+kd[i]: %f\n", h*kp[i]+kd[i]) 
                # wp.printf("wp.dot(J_kernel[0], -J_div_m_kernel[0]): %f\n", wp.dot(J_kernel[0], -J_div_m_kernel[0]))   
            d[i][j] = Vec3(d_ij[0], wp.asin(d_ij[1]), wp.asin(d_ij[2]))
            
    elif(con_type[i] == BODY_ROTATE_CONSTRAINT or con_type[i] == BODY_ROTATE_TARGET_CONSTRAINT):
            b1_i = body1_ind[i]
            b2_i = body2_ind[i]
            cf = getContactFrame(wp.transform_vector(q[b1_i][j], axis1[i]))
            w_kernel = Vec3()
            J_div_m_kernel = Mat36()
            J_kernel = Mat36()
            
            M_r = Vec3(I[b1_i][0],I[b1_i][1],I[b1_i][2])
            for k in range(3):
                n_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), cf[k])
                w_kernel[k] = wp.dot(n_l, wp.cw_div(n_l, M_r))
                rxnI = wp.transform_vector(q[b1_i][j], wp.cw_div(n_l, M_r))
                J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], 0.0, 0.0, 0.0)
                J_kernel[k] = Vec6(cf[k][0], cf[k][1], cf[k][2], 0.0, 0.0, 0.0)
            w1[i][j] = w_kernel
            J1[i][j] = J_kernel
            J_div_m1[i][j] = J_div_m_kernel
            
            M_r = Vec3(I[b2_i][0],I[b2_i][1],I[b2_i][2])
            for k in range(3):
                n_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), cf[k])
                w_kernel[k] = wp.dot(n_l, wp.cw_div(n_l, M_r))
                rxnI = wp.transform_vector(q[b2_i][j], wp.cw_div(n_l, M_r))
                J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], 0.0, 0.0, 0.0)
                J_kernel[k] = Vec6(cf[k][0], cf[k][1], cf[k][2], 0.0, 0.0, 0.0)
            w2[i][j] = w_kernel
            J2[i][j] = J_kernel
            J_div_m2[i][j] = J_div_m_kernel
            
            d_ij = cf * wp.cross(wp.transform_vector(q[b2_i][j], axis2[i]), wp.transform_vector(q[b1_i][j], axis1[i]))
            n1= wp.transform_vector(q[b1_i][j], b1[i])
            n2 = wp.transform_vector(q[b2_i][j], b2[i])
            d_target = wp.asin(wp.dot(cf[0], wp.cross(n2, n1)))
            if(wp.dot(n1, n2) < 0.0):
                d_target = wp.PI - d_target
            d_target = wp.atan2(wp.sin(d_target), wp.cos(d_target))  # map to [-pi, pi]
            if(con_type[i] == BODY_ROTATE_CONSTRAINT):
                if(d_target > limits[i][0] - ROTATE_OFFSET):
                    d_ij[0] = d_target - limits[i][0]
                    limit_signs[i] = -1.0
                elif(d_target < limits[i][1] + ROTATE_OFFSET):
                    d_ij[0] = d_target - limits[i][1]
                    limit_signs[i] = 1.0
                else:
                    d_ij[0] = 0.0
                    limit_signs[i] = 0.0
            elif(con_type[i] == BODY_ROTATE_TARGET_CONSTRAINT):
                d_J_div_m = J_div_m1[i][j] - J_div_m2[i][j]
                # dtheta_align = wp.cross(wp.transform_vector(q[b2_i][j], axis2[i]), wp.transform_vector(q[b1_i][j], axis1[i]))
                # dtheta_align_norm = wp.norm_l2(dtheta_align)
                # if dtheta_align_norm > EPS_SMALL:
                #     dq_align = wp.quat_from_axis_angle(dtheta_align / dtheta_align_norm, dtheta_align_norm)
                # else:
                #     dq_align = wp.quat_identity()
                # d_target = wp.dot(cf[0], wp.cross(wp.quat_rotate(dq_align * q[b_i][j].q, b2[i]), wp.transform_vector(q[b1_i][j], b1[i])))
                d_target = wp.asin(wp.dot(cf[0], wp.cross(wp.transform_vector(q[b2_i][j], b2[i]), wp.transform_vector(q[b1_i][j], b1[i]))))
                d_omega = wp.dot(J_kernel[0], phi[b1_i][j] - phi[b2_i][j])          
                a = h/(h*(h*kp[i]+kd[i])*wp.dot(J_kernel[0], d_J_div_m[0]) + 1.0)
                d_ij[0] = a * h * (kp[i]*((d_target - theta_target[i]) + d_omega * h) + kd[i]*(d_omega - oemga_target[i])) - d_omega * h
            d[i][j] = Vec3(d_ij[0], wp.asin(d_ij[1]), wp.asin(d_ij[2]))
        
@wp.kernel
def solveConstraintsJoint(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body1_ind: wp.array(dtype=INT_DATA_TYPE),
    body2_ind: wp.array(dtype=INT_DATA_TYPE),
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
    d: wp.array(dtype=Vec3, ndim=2),
    limit_signs: wp.array(dtype=FP_DATA_TYPE),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=Vec3, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2)
):
    j = wp.tid()

    for i in range(con_count):
        if(con_type[i] == WORLD_FIX_CONSTRAINT or con_type[i] == WORLD_ROTATE_CONSTRAINT or con_type[i] == WORLD_ROTATE_TARGET_CONSTRAINT):
            b_i = body2_ind[i]
            d_ij = d[i][j]
            phi_ij = phi[b_i][j]
            w_ij = w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias = phi_dt[b_i][j] / h
            J_ij = J2[i][j]
            J_div_m_ij = J_div_m2[i][j]
            for k in range(3):
                c = -wp.dot(J_ij[k], (phi_ij +bias)) + d_ij[k] / h
                # wp.printf("c: %f = %f + %f\n", c, wp.dot(J_ij[k], (phi_ij +bias)), d_ij[k] / h)
                # wp.printf("d_ij: %f \n", d_ij[k])
                # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
                # wp.printf("phi: %f %f %f %f %f %f\n", phi_ij[0], phi_ij[1], phi_ij[2], phi_ij[3], phi_ij[4], phi_ij[5])
                # wp.printf("Jijk: %f %f %f %f %f %f\n", J_ij[k][0], J_ij[k][1], J_ij[k][2], J_ij[k][3], J_ij[k][4], J_ij[k][5])
                dlambda = -c / w_ij[k]
                # wp.printf("dlambda: %f \n", dlambda)
                if(con_type[i] == WORLD_ROTATE_CONSTRAINT and k==0 and (lambdas_ij[0] + dlambda) * limit_signs[i] <= 0.0):
                    dlambda = -lambdas_ij[0]
                lambdas_ij[k] += dlambda
                phi_ij -= J_div_m_ij[k] * dlambda
            lambdas[i][j] = lambdas_ij
            phi[b_i][j] = phi_ij
        elif(con_type[i] == BODY_FIX_CONSTRAINT or con_type[i] == BODY_ROTATE_CONSTRAINT or con_type[i] == BODY_ROTATE_TARGET_CONSTRAINT):
            b1_i = body1_ind[i]
            b2_i = body2_ind[i]
            d_ij = d[i][j]
            phi1_ij = phi[b1_i][j]
            phi2_ij = phi[b2_i][j]
            w_ij = w1[i][j] + w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias1 = phi_dt[b1_i][j] / h
            bias2 = phi_dt[b2_i][j] / h
            J1_ij = J1[i][j]
            J_div_m1_ij = J_div_m1[i][j]
            J2_ij = J2[i][j]
            J_div_m2_ij = J_div_m2[i][j]
            for k in range(3):
                c = wp.dot(J1_ij[k], (phi1_ij +bias1)) - wp.dot(J2_ij[k], (phi2_ij +bias2)) + d_ij[k] / h
                # wp.printf("c: %f = %f - %f\n", c, wp.dot(J1_ij[k], (phi1_ij +bias1)), wp.dot(J2_ij[k], (phi2_ij +bias2)))
                # wp.printf("w_ij: %f \n", w_ij[k])
                # wp.printf("d_ij: %f \n", d_ij[k])
                # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
                # wp.printf("phi1: %f %f %f %f %f %f\n", phi1_ij[0], phi1_ij[1], phi1_ij[2], phi1_ij[3], phi1_ij[4], phi1_ij[5])
                # wp.printf("phi2: %f %f %f %f %f %f\n", phi2_ij[0], phi2_ij[1], phi2_ij[2], phi2_ij[3], phi2_ij[4], phi2_ij[5])
                # wp.printf("J1ijk: %f %f %f %f %f %f\n", J1_ij[k][0], J1_ij[k][1], J1_ij[k][2], J1_ij[k][3], J1_ij[k][4], J1_ij[k][5])
                # wp.printf("J2ijk: %f %f %f %f %f %f\n", J2_ij[k][0], J2_ij[k][1], J2_ij[k][2], J2_ij[k][3], J2_ij[k][4], J2_ij[k][5])
                dlambda = -c / w_ij[k]
                if(con_type[i] == BODY_ROTATE_CONSTRAINT and k==0 and (lambdas_ij[0] + dlambda) * limit_signs[i] <= 0.0):
                    dlambda = -lambdas_ij[0]
                # wp.printf("dlambda: %f \n", dlambda)
                lambdas_ij[k] += dlambda
                phi1_ij += J_div_m1_ij[k] * dlambda
                phi2_ij -= J_div_m2_ij[k] * dlambda
            lambdas[i][j] = lambdas_ij
            phi[b1_i][j] = phi1_ij
            phi[b2_i][j] = phi2_ij

import warp as wp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DataTypes import *
from utils import SIM_NUM, EPS_SMALL, VIA_POINT_MAX

BODY_MUSCLE_CONSTRAINT = wp.constant(9)

class MLP(nn.Module):
    """Same MLP as in your Physics.py"""
    def __init__(self, input_size=3, hidden_size=256, output_size=1, num_layers=5, activation='tanh'):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Create layers dynamically
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.layers = nn.ModuleList(layers)
        
        # Set activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = torch.tanh  # Default
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

@wp.kernel
def initConstraintsMuscle(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body_num: wp.array(dtype=INT_DATA_TYPE),
    body_inds: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    muscle_params: wp.array(dtype=FP_DATA_TYPE, ndim=2),  # [lMopt, lTslack, vMmax, alphaMopt, fMopt, beta, h, compliance]
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xls: wp.array(dtype=Vec3, ndim=2),
    h: FP_DATA_TYPE,
    # outputs
    alpha: wp.array(dtype=FP_DATA_TYPE),
    Js: wp.array(dtype=Vec6, ndim=3),
    J_div_ms: wp.array(dtype=Vec6, ndim=3),
    w: wp.array(dtype=FP_DATA_TYPE, ndim=2),
):
    i, j = wp.tid()
    if i >= con_count:
        return
    
    if con_type[i] == BODY_MUSCLE_CONSTRAINT and body_num[i] == 2:
        # Extract muscle parameters
        compliance = muscle_params[i][7]  # 1/fMopt
        alpha[i] = compliance / (h * h)
        
        b1_i = body_inds[i][0]  # fixed body
        b2_i = body_inds[i][1]  # moving body
        
        # Get positions of attachment points
        p1 = wp.transform_point(q[b1_i][j], xls[i][0])
        p2 = wp.transform_point(q[b2_i][j], xls[i][1])
        
        # Calculate current distance and direction
        dx = p1 - p2
        current_length = wp.length(dx)
        
        if current_length < EPS_SMALL:
            return
            
        # Normalized direction
        n = dx / current_length
        
        # Calculate weights (inverse mass matrix contributions)
        M_r1 = Vec3(I[b1_i][0], I[b1_i][1], I[b1_i][2])
        M_p1 = I[b1_i][3]
        M_r2 = Vec3(I[b2_i][0], I[b2_i][1], I[b2_i][2])
        M_p2 = I[b2_i][3]
        
        # Local coordinates of attachment points
        xl1 = xls[i][0]
        xl2 = xls[i][1]
        
        # Compute Jacobians (geometric relationship)
        n1_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), n)
        n2_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), -n)
        
        rxn1_l = wp.cross(xl1, n1_l)
        rxn2_l = wp.cross(xl2, n2_l)
        
        # Global Jacobians
        rxn1 = wp.transform_vector(q[b1_i][j], rxn1_l)
        rxn2 = wp.transform_vector(q[b2_i][j], rxn2_l)
        
        Js[i][j][0] = Vec6(rxn1[0], rxn1[1], rxn1[2], n[0], n[1], n[2])
        Js[i][j][1] = Vec6(rxn2[0], rxn2[1], rxn2[2], -n[0], -n[1], -n[2])
        
        # Inverse mass weighted Jacobians
        rxn1I = wp.transform_vector(q[b1_i][j], wp.cw_div(rxn1_l, M_r1))
        rxn2I = wp.transform_vector(q[b2_i][j], wp.cw_div(rxn2_l, M_r2))
        
        J_div_ms[i][j][0] = Vec6(rxn1I[0], rxn1I[1], rxn1I[2], n[0] / M_p1, n[1] / M_p1, n[2] / M_p1)
        J_div_ms[i][j][1] = Vec6(rxn2I[0], rxn2I[1], rxn2I[2], -n[0] / M_p2, -n[1] / M_p2, -n[2] / M_p2)
        
        # Compute effective mass
        w[i][j] = wp.dot(rxn1_l, wp.cw_div(rxn1_l, M_r1)) + 1.0 / M_p1 + \
                 wp.dot(rxn2_l, wp.cw_div(rxn2_l, M_r2)) + 1.0 / M_p2

@wp.kernel
def solveMuscleConstraints(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body_num: wp.array(dtype=INT_DATA_TYPE),
    body_inds: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    muscle_params: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    alpha: wp.array(dtype=FP_DATA_TYPE),
    Js: wp.array(dtype=Vec6, ndim=3),
    J_div_ms: wp.array(dtype=Vec6, ndim=3),
    w: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    # Neural network outputs (computed on CPU, passed to GPU)
    C_values: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    gradients: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    pennation_angles: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    # outputs
    lambdas: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
):
    j = wp.tid()
    
    for i in range(con_count):
        if con_type[i] == BODY_MUSCLE_CONSTRAINT and body_num[i] == 2:
            # Get constraint violation from neural network
            C = C_values[i][j]
            grad = gradients[i][j]
            penn = pennation_angles[i][j]
            
            # Add current constraint state (similar to spring bias)
            c = C + alpha[i] * lambdas[i][j]
            
            # Add Jacobian contributions (from current body corrections)
            for k in range(body_num[i]):
                b_i = body_inds[i][k]
                c += wp.dot(Js[i][j][k], phi[b_i][j])
            
            # Solve for lambda increment (your original denominator calculation)
            denominator = w[i][j] * grad * grad + alpha[i]
            dlambda = -c / denominator
            
            # Apply pennation to the correction (as in your implementation)
            dlambda_with_pennation = dlambda * wp.cos(penn)
            
            # Update lambda (framework requirement)
            lambdas[i][j] += dlambda_with_pennation
            
            # Apply position corrections with gradient scaling and pennation
            for k in range(body_num[i]):
                b_i = body_inds[i][k]
                phi[b_i][j] += J_div_ms[i][j][k] * dlambda_with_pennation * grad

class ConstraintsOnDevice():
    def __init__(self):
        self.con_count = wp.zeros(shape=SIM_NUM, dtype=INT_DATA_TYPE)

    def update(self, step):
        pass

    def computeC(self):
        pass
    
    def init(self):
        pass

    def solve(self):
        pass

    def solveVelocity(self):
        pass

class ConstraintsContact(ConstraintsOnDevice):
    def __init__(self, fixed, bodies_on_host, bodies_on_device, body_pairs):
        super().__init__()
        self.fixed = fixed
        self.bodies_on_host = bodies_on_host
        self.bodies = bodies_on_device
        self.body_pairs = body_pairs
        self.con_type = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=INT_DATA_TYPE)
        self.c = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.body_ind1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=INT_DATA_TYPE)
        self.body_ind2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=INT_DATA_TYPE)
        self.J1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.J2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.J_div_m1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.J_div_m2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.xl1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.xl2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.w1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.w2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.normal = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.d = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=FP_DATA_TYPE)
        self.mu = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=FP_DATA_TYPE)

    def update(self, step):
        self.con_count.zero_()
        for i in range(len(self.fixed)):
            for j in range(len(self.bodies_on_host)):
                collisionDetectionGroundCuboid(self.bodies_on_host[j].index,
                                                self.bodies.transform[j],
                                                self.bodies_on_host[j].shape.sides,
                                                self.bodies_on_host[j].mu,
                                                self.fixed[i].transform,
                                                self.fixed[i].shape.normal,
                                                self.con_count,
                                                self.con_type,
                                                self.body_ind2,
                                                self.d,
                                                self.normal,
                                                self.xl2,
                                                self.mu)
        for i in range(len(self.body_pairs)):
            body1, body2 = self.body_pairs[i]
            collisionDetectionCuboidCuboid(body1,
                                            self.bodies.transform[body1],
                                            self.bodies_on_host[body1].shape.sides,
                                            body2,
                                            self.bodies.transform[body2],
                                            self.bodies_on_host[body2].shape.sides,
                                            0.5 * (self.bodies_on_host[body1].mu + self.bodies_on_host[body2].mu),
                                            self.con_count,
                                            self.con_type,
                                            self.body_ind1,
                                            self.body_ind2,
                                            self.d,
                                            self.normal,
                                            self.xl1,
                                            self.xl2,
                                            self.mu)
            # print("con_count: ",self.con_count.numpy())
            # print("con_type: ",self.con_type.numpy())
            # print("d: ",self.d.numpy())
            # print("normal: ",self.normal.numpy())
            # print("xl1: ",self.xl1.numpy())
            # print("xl2: ",self.xl2.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass

    def init(self, h):
        wp.launch(
            kernel=initConstraintsContact,
            dim=(CONTACT_MAX, SIM_NUM),
            inputs=[
                self.con_count,
                self.con_type,
                self.body_ind1,
                self.body_ind2,
                self.bodies.transform,
                self.bodies.I,
                self.xl1,
                self.xl2,
                self.normal
            ],
            outputs=[
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2
            ],
            record_tape=False,
        )
        # print("J1: ",self.J1.numpy())
        # print("J_div_m1: ",self.J_div_m1.numpy())
        # print("w1: ",self.w1.numpy())
        # print("J2: ",self.J2.numpy())
        # print("J_div_m2: ",self.J_div_m2.numpy())
        # print("w2: ",self.w2.numpy())

    def solve(self, h):
        wp.launch(
            kernel=solveConstraintsContact,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body_ind1,
                self.body_ind2,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2,
                self.d,
                h,
                self.mu,
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            record_tape=False
        )
        wp.synchronize()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())

    def solveVelocity(self, h):
        self.solve(h)

class ConstraintsJoint(ConstraintsOnDevice):
    def __init__(self, joints, bodies_on_device):
        self.con_count = 2 * len(joints)
        self.bodies = bodies_on_device
        self.joints = joints

        body1_ind = np.zeros(shape=self.con_count, dtype=INT_DATA_TYPE)
        body2_ind = np.zeros(shape=self.con_count, dtype=INT_DATA_TYPE)
        constraint_type = np.empty(shape=self.con_count, dtype=INT_DATA_TYPE)
        xl1 = np.empty(shape=self.con_count, dtype=Vec3)
        xl2 = np.empty(shape=self.con_count, dtype=Vec3)
        axis1 = np.empty(shape=self.con_count, dtype=Vec3)
        axis2 = np.empty(shape=self.con_count, dtype=Vec3)
        b1 = np.empty(shape=self.con_count, dtype=Vec3)
        b2 = np.empty(shape=self.con_count, dtype=Vec3)
        kp = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        kd = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        limits = np.empty(shape=self.con_count, dtype=Vec2)
        ci = 0
        
        for i in range(len(joints)):
            if(joints[i].parent is None):
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                constraint_type[ci] = WORLD_FIX_CONSTRAINT
                ci +=1
                
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                limits[ci] = joints[i].limits
                if(joints[i].has_target):
                    constraint_type[ci] = WORLD_ROTATE_TARGET_CONSTRAINT
                    kp[ci] = joints[i].kp
                    kd[ci] = joints[i].kd
                else:
                    constraint_type[ci] = WORLD_ROTATE_CONSTRAINT
                ci +=1
            else:
                body1_ind[ci] = joints[i].parent
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                constraint_type[ci] = BODY_FIX_CONSTRAINT
                ci +=1
                
                body1_ind[ci] = joints[i].parent
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                limits[ci] = joints[i].limits
                if(joints[i].has_target):
                    constraint_type[ci] = BODY_ROTATE_TARGET_CONSTRAINT
                    kp[ci] = joints[i].kp
                    kd[ci] = joints[i].kd
                else:
                    constraint_type[ci] = BODY_ROTATE_CONSTRAINT
                ci +=1
                

        self.body1_ind = wp.array(body1_ind, dtype=INT_DATA_TYPE)
        self.body2_ind = wp.array(body2_ind, dtype=INT_DATA_TYPE)
        self.xl1 = wp.array(xl1, dtype=Vec3)
        self.xl2 = wp.array(xl2, dtype=Vec3)
        self.axis1 = wp.array(axis1, dtype=Vec3)
        self.axis2 = wp.array(axis2, dtype=Vec3)
        self.b1 = wp.array(b1, dtype=Vec3)
        self.b2 = wp.array(b2, dtype=Vec3)
        self.con_type = wp.array(constraint_type, dtype=INT_DATA_TYPE)
        self.kp = wp.array(kp, dtype=FP_DATA_TYPE)
        self.kd = wp.array(kd, dtype=FP_DATA_TYPE)
        self.limits = wp.array(limits, dtype=Vec2)
        self.theta_target = wp.empty(shape=self.con_count, dtype=FP_DATA_TYPE)
        self.omega_target = wp.empty(shape=self.con_count, dtype=FP_DATA_TYPE)
        self.limit_signs = wp.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        
        self.c = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.J1 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.J_div_m1 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.w1 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.J2 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.J_div_m2 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.w2 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.d = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)

    def update(self, step):
        theta_target = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        omega_target = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        ci = 0
        for i in range(len(self.joints)):
            if(self.joints[i].has_target):
                theta_target[ci+1] = self.joints[i].theta_target[step]
                omega_target[ci+1] = self.joints[i].omega_target[step]
            ci+=2
        self.theta_target = wp.array(theta_target, dtype=FP_DATA_TYPE)
        self.omega_target = wp.array(omega_target, dtype=FP_DATA_TYPE)

    def init(self, h):
        wp.launch(
            kernel=initConstraintsJoint,
            dim=(self.con_count, SIM_NUM),
            inputs=[
                self.con_count,
                self.con_type,
                self.body1_ind,
                self.body2_ind,
                self.bodies.phi,
                self.bodies.transform,
                self.bodies.I,
                self.xl1,
                self.xl2,
                self.axis1,
                self.axis2,
                self.b1,
                self.b2,
                self.theta_target,
                self.omega_target,
                self.kp,
                self.kd,
                self.limits,
                self.limit_signs,
                h,
            ],
            outputs=[
                self.d,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2
            ],
            record_tape=False,
        )
        # print("J1: ",self.J1.numpy())
        # print("J_div_m1: ",self.J_div_m1.numpy())
        # print("w1: ",self.w1.numpy())
        # print("J2: ",self.J2.numpy())
        # print("J_div_m2: ",self.J_div_m2.numpy())
        # print("w2: ",self.w2.numpy())
        # print("d: ",self.d.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass

    def solve(self, h):
        wp.launch(
            kernel=solveConstraintsJoint,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body1_ind,
                self.body2_ind,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2,
                self.d,
                self.limit_signs,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            record_tape=False
        )
        wp.synchronize()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())

    def solveVelocity(self, h):
        wp.launch(
            kernel=solveConstraintsJoint,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body1_ind,
                self.body2_ind,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2,
                self.d,
                self.limit_signs,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            record_tape=False
        )
        wp.synchronize()
        # print("lambdas: ",self.lambdas.numpy())
        
@wp.kernel
def initConstraintsMuscle_Simple(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body_inds: wp.array(dtype=INT_DATA_TYPE, ndim=2),  # [fixed_body_id, moving_body_id]
    muscle_params: wp.array(dtype=FP_DATA_TYPE, ndim=2),  # [lMopt, fMopt, compliance]
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    h: FP_DATA_TYPE,
    # outputs
    alpha: wp.array(dtype=FP_DATA_TYPE),
    w2: wp.array(dtype=FP_DATA_TYPE, ndim=2),  # Weight of moving particle
):
    i, j = wp.tid()
    if i >= con_count:
        return
    
    if con_type[i] == BODY_MUSCLE_CONSTRAINT:
        # Extract muscle parameters
        compliance = muscle_params[i][2]
        alpha[i] = compliance / (h * h)
        
        # Get moving body (second body in the pair)
        moving_body_id = body_inds[i][1]
        
        # Weight = 1/mass (for moving particle only)
        mass = 1.0 / I[moving_body_id][3]  # I[3] is 1/mass, so this gives mass
        w2[i][j] = 1.0 / mass if mass > 0.0 else 0.0

@wp.kernel
def solveMuscleConstraints_Simple(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body_inds: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    muscle_params: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    q: wp.array(dtype=Transform, ndim=2),
    q_prev: wp.array(dtype=Transform, ndim=2),
    alpha: wp.array(dtype=FP_DATA_TYPE),
    w2: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    gravity: Vec3,
    # Neural network outputs
    C_values: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    gradients: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    pennation_angles: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    h: FP_DATA_TYPE,
    # outputs
    phi: wp.array(dtype=Vec6, ndim=2),
):
    j = wp.tid()
    
    for i in range(con_count):
        if con_type[i] == BODY_MUSCLE_CONSTRAINT:
            fixed_body_id = body_inds[i][0]
            moving_body_id = body_inds[i][1]
            
            # Get positions (assuming position is in transform)
            fixed_pos = Vec3(q[fixed_body_id][j][0], q[fixed_body_id][j][1], q[fixed_body_id][j][2])
            moving_pos = Vec3(q[moving_body_id][j][0], q[moving_body_id][j][1], q[moving_body_id][j][2])
            
            # Calculate direction vector
            dx = fixed_pos - moving_pos
            dx_length = wp.length(dx)
            
            if dx_length < EPS_SMALL:
                continue
                
            n = dx / dx_length
            
            # Get constraint values from neural network
            C = C_values[i][j]
            grad = gradients[i][j]
            penn = pennation_angles[i][j]
            
            # Your original denominator calculation
            denominator = w2[i][j] * grad * grad + alpha[i]
            
            # Your original delta_lambda calculation (without pennation first)
            delta_lambda = -C / denominator
            
            # Apply pennation to final correction (as in your implementation)
            delta_lambda_with_penn = delta_lambda * wp.cos(penn)
            
            # Apply correction to moving particle only (your implementation)
            # Position correction: moving_particle.position += w2 * delta_lambda * grad * n
            correction = Vec3(
                w2[i][j] * delta_lambda_with_penn * grad * n[0],
                w2[i][j] * delta_lambda_with_penn * grad * n[1],
                w2[i][j] * delta_lambda_with_penn * grad * n[2]
            )
            
            # Apply to moving body only (phi represents position corrections)
            phi[moving_body_id][j] = Vec6(
                phi[moving_body_id][j][0],  # Keep rotational parts unchanged
                phi[moving_body_id][j][1],
                phi[moving_body_id][j][2],
                phi[moving_body_id][j][3] + correction[0],  # Add translational correction
                phi[moving_body_id][j][4] + correction[1],
                phi[moving_body_id][j][5] + correction[2]
            )

class ConstraintMuscle:
    def __init__(self, muscles, bodies_on_device, device='cpu', model_path='TrainedModels\Muscles\len_act_vel_penn0_model.pth'):
        self.con_count = len(muscles)
        self.bodies = bodies_on_device
        self.muscles = muscles
        self.device = device
        
        # Initialize PyTorch model
        self.model = MLP(input_size=3, hidden_size=128, output_size=1, num_layers=4, activation='tanh')
        
        # Load trained model if path provided
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print(f"Successfully loaded model from: {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
        else:
            print("No model path provided, using random weights")
        
        # Simplified constraint data - only what we need
        body_inds = np.zeros(shape=(self.con_count, 2), dtype=INT_DATA_TYPE)  # [fixed, moving]
        constraint_type = np.full(shape=self.con_count, fill_value=BODY_MUSCLE_CONSTRAINT, dtype=INT_DATA_TYPE)
        muscle_params_array = np.zeros(shape=(self.con_count, 3), dtype=FP_DATA_TYPE)  # [lMopt, fMopt, compliance]
        
        # Populate muscle data
        for i, muscle in enumerate(muscles):
            if len(muscle.bodies) >= 2:
                body_inds[i][0] = muscle.bodies[0]  # Fixed body
                body_inds[i][1] = muscle.bodies[1]  # Moving body
                
                # Simplified muscle parameters
                params = muscle.muscle_params
                muscle_params_array[i] = [
                    params.lMopt,
                    params.fMopt,
                    params.compliance
                ]
            
        # Convert to warp arrays
        self.body_inds = wp.array(body_inds, dtype=INT_DATA_TYPE)
        self.con_type = wp.array(constraint_type, dtype=INT_DATA_TYPE)
        self.muscle_params = wp.array(muscle_params_array, dtype=FP_DATA_TYPE)
        
        # Initialize constraint solution arrays
        self.alpha = wp.empty(shape=self.con_count, dtype=FP_DATA_TYPE)
        self.w2 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        self.d = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        
        # Neural network output arrays
        self.C_values = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        self.gradients = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        self.pennation_angles = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        
        # Current simulation time and activation
        self.current_time = 0.0
        self.current_activations = np.zeros(self.con_count)
        
        # Store previous transforms for velocity computation
        self.prev_transforms = None
        
    def update(self, t):
        """Update method called by framework"""
        self.current_time = t
        # Update activations (sinusoidal like your implementation)
        for i in range(self.con_count):
            self.current_activations[i] = np.sin(t * 20) / 2.0 + 0.5
        
    def init(self, h):
        """Initialize constraints - called once per timestep"""
        wp.launch(
            kernel=initConstraintsMuscle_Simple,
            dim=(self.con_count, SIM_NUM),
            inputs=[
                self.con_count,
                self.con_type,
                self.body_inds,
                self.muscle_params,
                self.bodies.transform,
                self.bodies.I,
                h
            ],
            outputs=[
                self.alpha,
                self.w2,
            ],
            record_tape=False,
        )
        
        # Store current transforms for velocity computation
        self.prev_transforms = self.bodies.transform.numpy().copy()
        
    def solve(self, h):
        """Solve constraints - called once per substep (matches your implementation)"""
        # Evaluate neural network with PyTorch
        self._evaluate_neural_networks(h)
        
        # Solve muscle constraints
        wp.launch(
            kernel=solveMuscleConstraints_Simple,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body_inds,
                self.muscle_params,
                self.bodies.transform,
                self.bodies.transform if self.prev_transforms is None else wp.array(self.prev_transforms, dtype=Transform),
                self.alpha,
                self.w2,
                Vec3(0.0, -9.8, 0.0),  # Gravity
                self.C_values,
                self.gradients,
                self.pennation_angles,
                h
            ],
            outputs=[
                self.bodies.phi,
            ],
            record_tape=False
        )
        wp.synchronize()
        
        # Update previous transforms for next substep
        self.prev_transforms = self.bodies.transform.numpy().copy()
        
    def _evaluate_neural_networks(self, h):
        """Evaluate neural networks - exactly like your implementation"""
        current_transforms = self.bodies.transform.numpy()
        
        C_values_cpu = np.zeros((self.con_count, SIM_NUM), dtype=np.float32)
        gradients_cpu = np.zeros((self.con_count, SIM_NUM), dtype=np.float32)
        pennation_cpu = np.zeros((self.con_count, SIM_NUM), dtype=np.float32)
        
        for i in range(self.con_count):
            muscle = self.muscles[i]
            if len(muscle.bodies) >= 2:
                fixed_body_id = muscle.bodies[0]
                moving_body_id = muscle.bodies[1]
                params = muscle.muscle_params
                
                for j in range(SIM_NUM):
                    # Get positions from transforms
                    fixed_pos = current_transforms[fixed_body_id, j][:3]
                    moving_pos = current_transforms[moving_body_id, j][:3]
                    
                    # Calculate muscle vector and length
                    dx = fixed_pos - moving_pos
                    current_length = np.linalg.norm(dx)
                    
                    if current_length < 1e-6:
                        continue
                        
                    # Calculate pennation angle (your implementation)
                    if current_length > params.h:
                        penn = np.arcsin(params.h / current_length)
                    else:
                        penn = np.pi / 2.0
                    pennation_cpu[i, j] = penn
                    
                    # Calculate relative velocity
                    if self.prev_transforms is not None:
                        prev_fixed_pos = self.prev_transforms[fixed_body_id, j][:3]
                        prev_moving_pos = self.prev_transforms[moving_body_id, j][:3]
                        prev_dx = prev_fixed_pos - prev_moving_pos
                        
                        relative_motion = dx - prev_dx
                        n = dx / current_length
                        relative_vel = np.dot(relative_motion, n) / h
                    else:
                        relative_vel = 0.0
                    
                    # Prepare neural network inputs (your implementation)
                    lMtilde = current_length / params.lMopt
                    activation = self.current_activations[i]
                    vMtilde = relative_vel / (params.lMopt * params.vMmax)
                    
                    # Neural network evaluation (your implementation)
                    inputs = torch.tensor([lMtilde, activation, vMtilde], 
                                        dtype=torch.float32, requires_grad=True)
                    inputs = inputs.reshape(1, -1)
                    
                    # Forward pass
                    C = self.model(inputs)
                    C_values_cpu[i, j] = C.item()
                    
                    # Compute gradient (your implementation)
                    grad = torch.autograd.grad(C, inputs, 
                                             grad_outputs=torch.ones_like(C),
                                             create_graph=False)[0]
                    gradients_cpu[i, j] = grad[0, 0].item()  # dC/d(lMtilde)
        
        # Transfer to GPU
        self.C_values = wp.array(C_values_cpu, dtype=FP_DATA_TYPE)
        self.gradients = wp.array(gradients_cpu, dtype=FP_DATA_TYPE)
        self.pennation_angles = wp.array(pennation_cpu, dtype=FP_DATA_TYPE)
        
    def solveVelocity(self, h):
        """Velocity solving - reuse position solving for now"""
        self.solve(h)
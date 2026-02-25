from peacsim.simulator_collab import SimulatorCollab
from peacsim.AQA_runtime_policy import AQARuntimePolicy
from peacsim.run_simulator import init_node_table, init_output
from peacsim.create_tables import init_job_table
import numpy as np
import pandas as pd
import random
from scipy.optimize import fsolve
import sys
import os
import configparser
import wandb
import multiprocessing
from peacsim.power_balancer import PowerBalancer
import json
from peacsim.denormalize_PR import denormalize_p_and_r_to_watts
from extract_power_metrics import calculate_tracking_error_distribution

"""
Authors: @HudsonReynolds @JoshBard @FatihAcun @EthanL3
Created 07/08/2024
PeacLab
"""

# def parallel_increment_gradient(args):
#     (optimizer, iteration, beta, rho, P_in, R_in, Pscale, Rscale, weights, simulator, process_index) = args
#     return optimizer.increment_gradient(iteration, beta, rho, P_in, R_in, Pscale, Rscale, weights, simulator)
softplus_at_zero = np.log(2)  # ln(1+e^x) = ln(2) when x = 0


class GradientDescentOptimizerCollab:
    def __init__(self, gradient_configs, experiment_configs, cluster_configs,
                 policy_configs, job_configs, output_dir, logger=None):
        self.gradient_configs = gradient_configs
        self.experiment_configs = experiment_configs
        self.cluster_configs = cluster_configs
        self.policy_configs = policy_configs
        self.job_configs = job_configs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.dr_program_variables = self.gradient_configs[0].dr_program_variables
        self.num_iterations = self.gradient_configs[0].gradient_driver_variables['num_iterations']
        self.num_data_centers = len(self.experiment_configs)
        self.dr_program_type = self.dr_program_variables['program_type']
        self.logger = logger

        if self.dr_program_type == 'EDR':
            self.Pbase = float(self.dr_program_variables['pbase'])
            self.piI = float(self.dr_program_variables['pii'])

        if logger == 'wandb':
            for dc_index in range(self.num_data_centers):
                configurations = {
                    "gradient_config": self.gradient_configs[dc_index].to_dict(),
                    "experiment_config": self.experiment_configs[dc_index].to_dict(),
                    "cluster_config": self.cluster_configs[dc_index].to_dict(),
                    "policy_config": self.policy_configs[dc_index].to_dict(),
                    "job_config": self.job_configs[dc_index].to_dict()
                }

                wandb_configs = {**gradient_configs[dc_index].gradient_driver_variables,
                                 **gradient_configs[dc_index].general_variables,
                                 **gradient_configs[dc_index].lagrange_variables,
                                 **experiment_configs[dc_index].to_dict(),
                                 'policy_name': policy_configs[dc_index].runtime_policy_name,
                                 'dr_program_type': self.dr_program_type,
                                 'utilization': experiment_configs[dc_index].utilization,
                                 'server_count': experiment_configs[dc_index].server_count,
                                 'workload_mix': job_configs[dc_index].workload_mix_name.split('/')[-1],
                                 'beta': gradient_configs[dc_index].gradient_driver_variables['beta'],
                                 'rho': gradient_configs[dc_index].gradient_driver_variables['rho']
                                 }
                workload_names = []
                for job_config in job_configs:
                    workload_names.append(job_config.workload_mix_name.split('/')[-1].split('_')[0].split('.')[0])
                workload_names = "_".join(workload_names)
                wandb.init(project='Conductor++', config=wandb_configs, name=f"collab_{workload_names}")

        print("saving optimization configurations in", output_dir + "configurations.json")
        with open(self.output_dir + "configurations.json", "w") as file:
            json.dump(configurations, file, indent=4)

        if gradient_configs[0].use_lagrange:
            self.use_lagrange = True

            # self.qos_lagrange_list = []
            # self.tracking_lagrange_list = []
            # for dc_index in range(len(gradient_configs)):
            #     self.qos_lagrange_list.append(gradient_configs[dc_index].qos_lagrange)
            #     self.tracking_lagrange_list.append(gradient_configs[dc_index].tracking_lagrange)

        # Initialize augmented Lagrangian parameters (separate from regular Lagrange)
        if hasattr(gradient_configs[0], 'use_augmented_lagrange') and gradient_configs[0].use_augmented_lagrange:
            # Read initial values from config
            self.use_lagrange = True
            self.c_k_list = [gradient_configs[dc_index].c_k for dc_index in range(len(gradient_configs))]
            self.c_t_list = [gradient_configs[dc_index].c_t for dc_index in range(len(gradient_configs))]
            self.c_k_rate_list = [gradient_configs[dc_index].c_k_rate for dc_index in range(len(gradient_configs))]
            self.c_t_rate_list = [gradient_configs[dc_index].c_t_rate for dc_index in range(len(gradient_configs))]
            # Initialize mu_t and mu_j_values for augmented Lagrangian (always initialize)
            self.mu_t_list = [1] * len(gradient_configs)
            self.mu_j_values_list = [[1] * len(self.job_configs[dc_index].all_max_job_power) for dc_index in
                                    range(len(gradient_configs))]
        
        else:
            # Default values if not in config
            self.c_k_list = [1] * len(gradient_configs)
            self.c_t_list = [1] * len(gradient_configs)
            self.c_k_rate_list = [2] * len(gradient_configs)  # Default rate is 2
            self.c_t_rate_list = [2] * len(gradient_configs)  # Default rate is 2

            # self.qos_lagrange_list = []
            # self.tracking_lagrange_list = []
            # for dc_index in range(len(gradient_configs)):
            #     self.qos_lagrange_list.append(gradient_configs[dc_index].qos_lagrange)
            #     self.tracking_lagrange_list.append(gradient_configs[dc_index].tracking_lagrange)

    def calculate_delay_prob(self, fit, sim_hour, J, job_table, QoS_constraint, min_execution_time, dc_index):
        fitdown, fitup = self.gradient_configs[dc_index].calculate_delay_prob_variables.values()
        mean = {}
        delay_prob = []
        thetas = []
        if fit:
            with open(self.output_dir + 'dc_' + str(dc_index) + '/thetaStars.csv', 'r') as f:
                for line in f:
                    numbers = line.split(',')
                    for i in numbers:
                        thetas.append(float(i))
        # print('Thetas: ' + str([('%.5f' % x) for x in thetas]))
        for ijob in range(J):
            first_hour = 1 if sim_hour > 2 else 0  # Skip hour 0: queues are initially empty. perf is biased positively
            # there. Including hour 0 is better than running a single hour, so only do that if there are 3+ hours.
            last_hour = (
                    sim_hour - 1) if sim_hour > 2 else sim_hour  # ignore the final hour if there are enough hours to
            # trim. The last hour has draining jobs which biases our perf positively.
            this_job_finished = job_table[(job_table['job_type_id'] == ijob)
                                          & (job_table['end_time'] != -1)
                                          & (job_table['arrival_time'] > first_hour * 3600)]
            delay = list(this_job_finished['end_time'] - this_job_finished['arrival_time'])  # Sojourn time
            job_should_finish = job_table[(job_table['job_type_id'] == ijob)
                                          & (job_table['arrival_time'] + min_execution_time[ijob] < sim_hour * 3600)
                                          & (job_table['end_time'] == -1)
                                          & (job_table['arrival_time'] > first_hour * 3600)
                                          & (job_table['arrival_time'] <= last_hour * 3600)]
            if len(job_should_finish) > 0:
                delay = delay + [sim_hour * 3600 - x for x in
                                 list(job_should_finish['arrival_time'])]  # Use true sim_hour here
                # since we're estimating with start time as "now" at the end of the sim.
            delay = [x - min_execution_time[ijob] for x in delay]  # minus min execution time.
            sort_scaled = np.sort(delay)
            prob = np.linspace(0, 1, len(sort_scaled))  # the CDF.
            alphas = []
            delay_prob_added = False
            for idx, iprob in enumerate(prob):
                if sort_scaled[idx] > 1 * QoS_constraint[ijob] * min_execution_time[ijob] and not delay_prob_added:
                    delay_prob.append(1 - iprob)
                    delay_prob_added = True
                if idx == len(prob) - 1 and not delay_prob_added:
                    delay_prob.append(0)
                    delay_prob_added = True
                if fit and iprob >= fitdown and iprob <= fitup:
                    this_scaled = sort_scaled[idx]
                    this_alpha = (1 - iprob) / np.exp(- this_scaled * thetas[ijob])
                    if this_alpha <= 0:
                        print('alpha: %f' % this_alpha)
                    alphas.append(this_alpha)
            if not delay_prob_added:  # no job for this type.
                delay_prob.append(0)
            mean[ijob] = np.mean(alphas)
        return mean, delay_prob

    def init_gradient(self, simulator, sim_hour, job_power_max, job_size, rho, prob, dc_index):

        gamma, kwh, qos_threshold, eta, tau, ysigma, Tst, mst, psi1, psi2, tracking_err_constraint, dEdR = (
            self.gradient_configs[dc_index].calculate_gradient_variables.values())  # Read in initial values

        runtime_policy = simulator._runtime_policy  # Read simulator's values
        # job_config = simulator._job_config
        Pbar = runtime_policy._P
        R = runtime_policy._R
        weightedP = runtime_policy.weighted_average_power
        weights = runtime_policy.weights
        N = simulator._server_count
        J = simulator._job_type_count
        QoS_constraint = list(self.job_configs[dc_index].all_job_qos_constraints.values())
        min_execution_time = np.array(list(self.job_configs[dc_index].all_min_execution_time.values()))
        max_execution_time = list(self.job_configs[dc_index].all_max_execution_time.values())
        pidle = simulator._idle_power

        delta = qos_threshold  # Initial calculations
        H = sim_hour * 3600
        piP = piR = piE = 0.1 / kwh
        if self.dr_program_type == 'EDR':
            self.piI /= kwh
        dEdP = piE * H  # for alg2.
        # pavg = np.mean(job_power_max)  # Change from this to match old
        pavg = sum(job_power_max) / len(job_power_max)
        pst = pavg
        nmu = (Pbar - pidle * N) / (weightedP - pidle)
        nsigma = ysigma * R / (weightedP - pidle)
        dMudP = 1 / (weightedP - pidle)
        dSigmadR = ysigma / (weightedP - pidle)
        # Lambda = eta * N / J / min_execution_time / np.array(job_size)  # Change this to be a list or have more precision to match old
        Lambda = [eta * N / J / x / job_size[idx] for idx, x in enumerate(min_execution_time)]
        D, dMudw, dSigmadw, K0 = [], [], [], []

        for itype in range(J):
            D.append(QoS_constraint[itype] * min_execution_time[itype])  # only this T is not multiplied by job_size
            dMudw.append(-(Pbar - pidle * N) / (weightedP - pidle) ** 2 * (job_power_max[itype]))
            dSigmadw.append(-ysigma * R / (weightedP - pidle) ** 2 * (job_power_max[itype]))
            K0.append(np.log(1 + np.exp(
                rho * (prob[itype] - delta))) - softplus_at_zero)  # Softplus function per job type's QoS deg prob

        return (gamma, kwh, qos_threshold, eta, tau, ysigma, Tst, mst, psi1, psi2, tracking_err_constraint, dEdP, dEdR,
                Pbar, R, weightedP, weights, N, J, QoS_constraint, min_execution_time, max_execution_time, delta, H,
                piP,
                piR, piE, pavg, pidle, pst, nmu, nsigma, dMudP, dSigmadR, Lambda, D, dMudw, dSigmadw, K0)

    def d_cost_error_alg3(self, R, Pbar, pidle, N, sum_expr, piE, H, psi1, psi2, tracking_err_constraint):
        numerator = (N * pidle - Pbar + sum_expr)
        err_estimated = numerator / R
        # sigmoid_term = np.exp(psi2 * (err_estimated - tracking_err_constraint)) / (
        #         1 + np.exp(psi2 * (err_estimated - tracking_err_constraint)))

        derivative = H * piE * psi1 * psi2 * numerator * np.abs(err_estimated) / (
                R * (np.exp(psi2 * (numerator + R * tracking_err_constraint) / R) + 1)
        )
        return derivative

    def d_cost_error_dR(self, R, bar_epsilon_estimation, psi1, psi2, tracking_err_constraint):
        z = psi2 * (bar_epsilon_estimation / R - tracking_err_constraint)
        sigmoid = np.exp(z) / (1.0 + np.exp(z))
        return -psi1 * psi2 * bar_epsilon_estimation / (R ** 2) * sigmoid

    def d_cost_error_dPbar(self, R, err_estimated, psi1, psi2, tracking_err_constraint):
        z = psi2 * (err_estimated - tracking_err_constraint)
        sigmoid = np.exp(z) / (1.0 + np.exp(z))
        return psi1 * psi2 / R * sigmoid

    def d_cost_augmented_error_dR(self, R, bar_epsilon_estimation, psi1, psi2, tracking_err_constraint, penalty_term):
        base_dEdR = self.d_cost_error_dR(R, bar_epsilon_estimation, psi1, psi2, tracking_err_constraint)

        if penalty_term > 0:
            return penalty_term * base_dEdR
        else:
            return 0

    def d_cost_augmented_error_dPbar(self, R, err_estimated, psi1, psi2, tracking_err_constraint, penalty_term):
        base_dEdP = self.d_cost_error_dPbar(R, err_estimated, psi1, psi2, tracking_err_constraint)
        if penalty_term > 0:
            return penalty_term * base_dEdP
        else:
            return 0

    def get_derivative_coeffs_of_qos_penalty(self, qos_penalty_terms):
        derivative_coeffs = []
        for j in range(len(qos_penalty_terms)):
            penalty_term = qos_penalty_terms[j]
            if penalty_term > 0:
                derivative_coeffs.append(penalty_term)
            else:
                derivative_coeffs.append(0)

        return derivative_coeffs

    def calculate_gradient(self, iteration, beta, rho, sim_hour, simulator, alpha, delay_prob, job_power_max, job_size,
                           dc_index, c_k, c_t, mu_t, mu_j_values):

        (
            gamma, kwh, qos_threshold, eta, tau, ysigma, Tst, mst, psi1, psi2, tracking_err_constraint, dEdP, dEdR,
            Pbar, R,
            weightedP, weights, N, J, QoS_constraint, min_execution_time, max_execution_time, delta, H, piP, piR, piE,
            pavg, pidle, pst, nmu, nsigma, dMudP, dSigmadR, Lambda, D, dMudw, dSigmadw, K0) = (
            self.init_gradient(simulator, sim_hour, job_power_max, job_size, rho, delay_prob, dc_index))

        expTable = pd.read_csv(self.output_dir + 'dc_' + str(dc_index) + '/power_trace.csv')
        err = expTable['trackingError'][1:]
        abserr = [abs(float(x)) for x in err.values]
        abserrSort = np.sort(abserr)
        errorAt90 = abserrSort[int(0.9 * len(abserrSort))]
        print('Error at 90%% percentile: %.4f' % errorAt90)

        ## From AQA paper equation 25-29
        bar_epsilon_estimation = (Pbar - np.sum(
            Lambda * job_size * min_execution_time * (np.array(job_power_max) - pidle)) - pidle * N)
        err_estimated = (1 / R) * bar_epsilon_estimation
        err_estimated =  errorAt90

        cost_error_alg2 = piE * H * R * abs(err_estimated)  # a first-order estimate.

        # Cost calculations
        cost_power = 0
        if self.dr_program_type == "EDR":
            cost_power = (piP * Pbar - self.piI * (self.Pbase - Pbar)) * H
        elif self.dr_program_type == "RSR":
            cost_power = (piP * Pbar - piR * R) * H + cost_error_alg2

        # cumulative_target = 0
        # cumulative_real_power = 0
        # cumulative_R = 0
        # for i in range(self.num_data_centers):
        #     power_trace = pd.read_csv(self.output_dir + 'dc_' + str(i) + '/power_trace.csv')
        #     R_i = pd.read_csv(self.output_dir + 'dc_' + str(i) + '/PRtable.csv')['R'][0]
        #     real_sum, target_power = power_trace['realSum'], power_trace['target']
        #     cumulative_target += target_power
        #     cumulative_real_power += real_sum
        #     cumulative_R += R_i
        #
        # collab_tracking_error_series = calculate_tracking_error_distribution(
        #     pd.DataFrame({'realSum': cumulative_real_power, 'target': cumulative_target}), cumulative_R)
        #
        # err_estimated = collab_tracking_error_series.mean()

        # cost_error_alg3 = cost_error_alg2 * (
        #         1 + psi1 * np.log(1 + np.exp(psi2 * (err_estimated - tracking_err_constraint))))

        tracking_error_cost_h = psi1 * np.log(
            1 + np.exp(psi2 * (err_estimated - tracking_err_constraint))) - softplus_at_zero
        cost_qos = beta * sum(K0)  # change from np.sum(K0) to match old
        costStandby = 0  # - gamma * (N - nmu) * pidle * H / (pst * Tst * mst)

        qos_costs_gj_max_squared = []
        qos_costs_gj = K0
        qos_penalty_terms = []
        for index, g_j in enumerate(K0):
            qos_penalty_terms.append(mu_j_values[index] + c_k * g_j)
            qos_costs_gj_max_squared.append(max(0, mu_j_values[index] + c_k * g_j) ** 2)

        qos_derivative_coeffs = self.get_derivative_coeffs_of_qos_penalty(qos_penalty_terms)

        tracking_penalty_term = mu_t + c_t * tracking_error_cost_h
        tracking_cost_max_squared = max(0, tracking_penalty_term) ** 2

        qos_term_full = (1 / (2 * c_k)) * sum(np.array(qos_costs_gj_max_squared) - np.array(mu_j_values) ** 2)

        tracking_term_full = (1 / (2 * c_t)) * (tracking_cost_max_squared - mu_t ** 2)

        augmented_lagrangian_cost = (cost_power
                                     + qos_term_full
                                     + tracking_term_full
                                     )

        cost_error = tracking_term_full  # the err in use. alg2 just scales w/ err. alg3 also penalizes constraint violations
        cost_qos = qos_term_full

        KP, KR, Kw = [], [], []
        Kw, dThetadw, dtheta0dw = {}, {}, {}
        for itype in range(J):
            eq = lambda x: (
                    Lambda[itype] * (np.expm1(min_execution_time[itype] * job_size[itype] * x)) - nmu * x * weights[
                itype] +
                    0.5 * nsigma ** 2 * tau * x ** 2 * weights[itype] ** 2)
            theta0 = fsolve(
                eq,
                8 * (nmu * weights[itype] - Lambda[itype] * min_execution_time[itype] * job_size[itype])
                / (Lambda[itype] * min_execution_time[itype] ** 2 * job_size[itype] ** 2
                   + nsigma ** 2 * tau * weights[itype] ** 2))[0]  # actually theta_0,j

            thetaStar = Lambda[itype] * (np.exp(theta0 * min_execution_time[itype] * job_size[itype]) - 1)
            under = (Lambda[itype] * np.exp(theta0 * min_execution_time[itype] * job_size[itype]) * min_execution_time[
                itype] *
                     job_size[itype] - nmu * weights[itype] + nsigma ** 2 * tau * theta0 * weights[itype] ** 2)
            K2 = Lambda[itype] * np.exp(theta0 * min_execution_time[itype] * job_size[itype]) * min_execution_time[
                itype] * job_size[itype]
            dtheta0dP = dMudP * theta0 * weights[itype] / under

            dThetadR = 0
            if self.dr_program_type == "RSR":
                dtheta0dR = -nsigma * tau * dSigmadR * theta0 ** 2 * weights[itype] ** 2 / under
                dThetadR = K2 * dtheta0dR
            dThetadP = K2 * dtheta0dP

            for jtype in range(J):
                xi = float(itype == jtype)
                dtheta0dw[(itype, jtype)] = (dMudw[jtype] * theta0 * weights[itype] + nmu * theta0 * xi - nsigma * tau *
                                             dSigmadw[jtype] *
                                             theta0 ** 2 * weights[itype] ** 2 - nsigma ** 2 * tau * theta0 ** 2 *
                                             weights[
                                                 itype] * xi) / under
                dThetadw[(itype, jtype)] = K2 * dtheta0dw[(itype, jtype)]
            K1 = (1 - 1 / (1 + np.exp(rho * (delay_prob[itype] - delta)))) * rho * delay_prob[itype] * (-D[itype])
            KP.append(K1 * dThetadP)
            if self.dr_program_type == "RSR":
                KR.append(K1 * dThetadR)

            for jtype in range(J):
                Kw[(itype, jtype)] = K1 * dThetadw[(itype, jtype)]

        # Compute gradients
        # dCdw = [self.qos_lagrange_list[dc_index] * beta * sum(Kw[(jtype, itype)] for jtype in range(J)) +
        #         gamma * pidle * H / (pst * Tst * mst) * dMudw[itype] for itype in range(J)]
        dCdw = [beta * sum(qos_derivative_coeffs[jtype] * Kw[(jtype, itype)] for jtype in range(J)) +
                gamma * pidle * H / (pst * Tst * mst) * dMudw[itype] for itype in range(J)]

        dCdR, dCdP = 0, 0

        # exponent = -psi2 * (1 / R - tracking_err_constraint)
        # softplus_derivative = 1 / (1 + np.exp(exponent))
        # dEdR = -piE * H * psi2 * (1/(R**2)) * softplus_derivative
        # dEdR = self.d_cost_error_alg3(R, Pbar, pidle, N, np.sum(Lambda * job_size * min_execution_time *
        #                                                         (np.array(job_power_max) - pidle)),
        #                               piE, H, psi1, psi2, tracking_err_constraint)

        #########################  Augmented Lagrangian with Softplus constraints  #########################
        dEdR = self.d_cost_augmented_error_dR(
            R=R,
            bar_epsilon_estimation=bar_epsilon_estimation,
            psi1=psi1,
            psi2=psi2,
            tracking_err_constraint=tracking_err_constraint,
            penalty_term=tracking_penalty_term
        )
        dEdP = self.d_cost_augmented_error_dPbar(R=R,
                                                 err_estimated=err_estimated,
                                                 psi1=psi1,
                                                 psi2=psi2,
                                                 tracking_err_constraint=tracking_err_constraint,
                                                 penalty_term=tracking_penalty_term
                                                 )

        # numerator = (N * pidle - Pbar + np.sum(Lambda * job_size * min_execution_time *
        #                                        (np.array(job_power_max) - pidle)))
        # err_estimated = numerator / R
        #
        # dEdP = -H * piE * psi1 * psi2 * (R * (R * (np.exp(psi2 * (numerator + R * tracking_err_constraint) / R) + 1)
        #                                       + psi2 * numerator) * np.abs(err_estimated) ** 2
        #                                  + (np.exp(psi2 * (numerator + R * tracking_err_constraint) / R) + 1)
        #                                  * numerator * (numerator - R * psi2 * np.abs(err_estimated) ** 2)) / \
        #        (R ** 3 * (np.exp(psi2 * (numerator + R * tracking_err_constraint) / R) + 1) ** 2 * np.abs(
        #            err_estimated))

        # dEdR = 0
        if self.dr_program_type == "RSR":
            for j in range(len(KP)):
                KP[j] = qos_derivative_coeffs[j] * KP[j]
                KR[j] = qos_derivative_coeffs[j] * KR[j]

            dCdP = piP * H + (sum(KP) + gamma * pidle * H / (pst * Tst * mst) * dMudP) \
                   + dEdP
            dCdR = -piR * H + (sum(KR)) + dEdR

            # dCdP = piP * H + self.qos_lagrange_list[dc_index] * (beta * sum(KP) + gamma * pidle * H / (pst * Tst * mst) * dMudP) \
            #        + self.tracking_lagrange_list[dc_index] * dEdP
            # dCdR = -piR * H + self.qos_lagrange_list[dc_index] * (beta * sum(KR)) + self.tracking_lagrange_list[dc_index] * dEdR

        elif self.dr_program_type == "EDR":
            dCdP = (piP + self.piI) * H + beta * sum(KP) + gamma * pidle * H / (pst * Tst * mst) * dMudP + dEdP

        dCd_qos_lagrange = 0  # cost_qos
        dCd_tracking_lagrange = 0  # cost_error

        # cost_qos *= self.qos_lagrange_list[dc_index]
        # cost_error *= self.tracking_lagrange_list[dc_index]

        ##### Update dual parameters #####
        # Update mu_j_values (create a copy to avoid modifying the original list)
        new_mu_j_values = mu_j_values.copy()
        any_qos_violation = False
        for index in range(len(mu_j_values)):
            new_mu_j_values[index] = max(0, qos_penalty_terms[index])
            if qos_costs_gj[index] > 0:
                any_qos_violation = True

        # Update mu_t
        new_mu_t = max(0, tracking_penalty_term)
        tracking_violation = tracking_error_cost_h > 0

        return dCdP, dCdR, dCdw, dCd_qos_lagrange, dCd_tracking_lagrange, cost_power, cost_error, cost_qos, costStandby, any_qos_violation, tracking_violation, new_mu_j_values, new_mu_t

    def get_thetas(self, utilization, server_count, weights, all_max_job_power, idle_power, all_job_min_execution_time,
                   job_type_count, all_job_size, P, R, dc_index):
        ysigma = 0.4
        tau = 46  # this number is the time scale where ISO signal becomes a true normal process.
        occupancy = [1 / job_type_count] * job_type_count
        Lambda = [x * utilization * server_count / all_job_size[i] / all_job_min_execution_time[i]
                  for i, x in enumerate(occupancy)]
        nmu = (P - idle_power * server_count) / (
                np.dot(all_max_job_power, weights) - idle_power)  # eqn 21
        nsigma = ysigma * R / (np.dot(all_max_job_power, weights) - idle_power)
        theta_stars = []
        for i in range(job_type_count):
            # (eq 23) (eqn 20)
            eq = lambda x: Lambda[i] * (np.expm1(all_job_size[i] * all_job_min_execution_time[i] * x)) - nmu * x * \
                           weights[
                               i] + 0.5 * nsigma ** 2 * tau * x ** 2 * weights[i] ** 2
            theta0 = fsolve(eq, nmu / (nsigma ** 2 * tau * weights[i]) / 1000)[0]
            # print(f'====== fsolve[{i}](eq, {nmu/(nsigma**2*tau*self.weights[i])/1000}) = {theta0}')
            # print(f'nmu={nmu}  --  numpy.expm1({jobSize[i]*minExecutionTime[i]*nmu/(nsigma**2*tau*self.weights[i])})')
            if theta0 < nmu / (nsigma ** 2 * tau * weights[i]):
                theta_star = nmu * theta0 * weights[i] - 0.5 * nsigma ** 2 * theta0 ** 2 * weights[i] ** 2 * tau
            else:
                print('Should check thetaStar.')
                theta_star = nmu ** 2 / (2 * nsigma ** 2 * tau)
            theta_stars.append(theta_star)

        print('ThetaStar: ' + str([('%f' % x) for x in theta_stars]))
        with open(self.output_dir + 'dc_' + str(dc_index) + '/thetaStars.csv', 'w') as ftheta:
            ftheta.write(','.join(['%f' % x for x in theta_stars]))
            ftheta.write('\n')

    def run_simulator(self, iteration, P_ratio, R_ratio, weights,
                      qos_values, power_assignments, current_time_shared, dc_index,
                      assignment_condition, power_balancer_condition, qos_write_lock):

        job_table = init_job_table(experiment_config=self.experiment_configs[dc_index],
                                   job_config=self.job_configs[dc_index])
        node_table = init_node_table(experiment_config=self.experiment_configs[dc_index])

        avg_max_job_power = sum(self.job_configs[dc_index].all_max_job_power.values()) / len(
            self.job_configs[dc_index].all_max_job_power)
        avg_num_servers = self.experiment_configs[dc_index].server_count * self.experiment_configs[dc_index].utilization
        avg_idle_servers = self.experiment_configs[dc_index].server_count - avg_num_servers
        P_ratio = round(P_ratio, 6)
        R_ratio = round(R_ratio, 6)
        self.policy_configs[dc_index]._P = int(P_ratio * (avg_max_job_power * avg_num_servers +
                                                          self.experiment_configs[
                                                              dc_index].idle_power * avg_idle_servers))

        self.policy_configs[dc_index]._R = int(
            R_ratio * (avg_max_job_power * self.experiment_configs[dc_index].server_count - (
                    self.experiment_configs[dc_index].idle_power * self.experiment_configs[dc_index].server_count)) / 2)

        policy_mapping = {'AQA': (
            AQARuntimePolicy,
            (self.policy_configs[dc_index], self.experiment_configs[dc_index], self.job_configs[dc_index], node_table,
             job_table))
            , 'newAQA': ()}
        policy_class, constructor_args = policy_mapping[self.policy_configs[dc_index].runtime_policy_name]
        runtime_policy = policy_class(*constructor_args)
        simulator = SimulatorCollab(job_table, node_table, runtime_policy, self.experiment_configs[dc_index],
                                    self.job_configs[dc_index],
                                    self.cluster_configs, self.output_dir + 'dc_' + str(dc_index) + "/",
                                    dc_index=dc_index,
                                    qos_values=qos_values,
                                    power_assignments=power_assignments,
                                    current_time_shared=current_time_shared,
                                    assignment_condition=assignment_condition,
                                    power_balancer_condition=power_balancer_condition,
                                    qos_write_lock=qos_write_lock)
        if iteration != 0:
            totalWeight = sum(weights)
            if totalWeight != 1:
                offset = (1 - totalWeight) / len(weights)
                for i in range(len(weights)):
                    weights[i] += offset
            if any(x < -0.01 for x in weights):
                print('Negative weight detected. Exit.')
                sys.exit(1)
            for key, new_value in zip(simulator._runtime_policy.weights.keys(), weights):
                simulator._runtime_policy.weights[key] = weights[key]
        simulator._runtime_policy.weighted_average_power = sum(
            simulator._runtime_policy.weights[job_type] * self.job_configs[dc_index].all_max_job_power[job_type] for
            job_type in
            simulator._runtime_policy.weights)

        print("P:", simulator._runtime_policy._P)
        print("R:", simulator._runtime_policy._R)
        print("Weights:", list(simulator._runtime_policy.weights.values()))
        # init_output(self.output_dir, self.experiment_configs[dc_index], self.policy_configs[dc_index])

        self.get_thetas(self.experiment_configs[dc_index].utilization,
                        self.experiment_configs[dc_index].server_count,
                        np.asarray(list(simulator._runtime_policy.weights.values())),
                        np.asarray(list(self.job_configs[dc_index].all_max_job_power.values())),
                        self.experiment_configs[dc_index].idle_power,
                        np.asarray(list(self.job_configs[dc_index].all_min_execution_time.values())),
                        self.job_configs[dc_index].job_type_count,
                        np.asarray(list(self.job_configs[dc_index].all_job_size.values())),
                        self.policy_configs[dc_index].P,
                        self.policy_configs[dc_index].R,
                        dc_index)
        print("...... Executing simulation ......")
        simulator.run()
        print("...... Simulation finished ......")
        return simulator

    def increment_gradient(self,
                           iteration,
                           P_in,
                           R_in,
                           Pscale,
                           Rscale,
                           weights,
                           weighted_avg_power,
                           dc_index,
                           qos_values,
                           power_assignments,
                           current_time_shared,
                           results_list,
                           assignment_condition,
                           power_balancer_condition,
                           qos_write_lock):

        (beta, rho, w_step_up, w_step_down, PR_up, PR_down, wmin, kappa, phi, limit_decrease, num_iterations) = (
            self.gradient_configs[dc_index].gradient_driver_variables.values())
        random.seed(self.experiment_configs[dc_index].random_seed + iteration)
        np.random.seed(self.experiment_configs[dc_index].random_seed + iteration)

        if iteration == 0:
            simulator = self.run_simulator(iteration, P_in, R_in, weights,
                                           qos_values, power_assignments, current_time_shared, dc_index,
                                           assignment_condition, power_balancer_condition, qos_write_lock)
        else:
            N = self.experiment_configs[dc_index].server_count
            P = P_in
            R = min(R_in, N * weighted_avg_power - P_in)  # Don't bid more power than we can reach up
            R = min(R_in, P_in - self.experiment_configs[
                dc_index].idle_power * N)  # Don't bid more power than we can reach down
            R = max(0, R_in)  # Don't bid negative capacity
            print(f'Batch sample before sim: P: {P}, Pscale: {Pscale}, R: {R}, Rscale: {Rscale}')
            simulator = self.run_simulator(iteration, P / Pscale, R / Rscale, weights,
                                           qos_values, power_assignments, current_time_shared, dc_index,
                                           assignment_condition, power_balancer_condition, qos_write_lock)

        # job_config = simulator._job_config
        QoS_constraint = list(self.job_configs[dc_index].all_job_qos_constraints.values())
        min_execution_time = list(self.job_configs[dc_index].all_min_execution_time.values())
        max_job_power = list(self.job_configs[dc_index].all_max_job_power.values())
        job_size = np.array(list(self.job_configs[dc_index].all_job_size.values()))
        P = simulator._runtime_policy._P
        R = simulator._runtime_policy._R
        w = list(simulator._runtime_policy.weights.values())
        N = simulator._server_count
        J = self.job_configs[dc_index].job_type_count
        job_table = simulator._job_table
        sim_hour = simulator._simulation_duration // 3600
        if iteration == 0:
            P_on_avg = sum(max_job_power) / float(len(max_job_power))
            N_on_avg = N * simulator._utilization
            N_idle_avg = N - N_on_avg
            Pscale_it = P_on_avg * N_on_avg + simulator._idle_power * N_idle_avg
            Pscale = Pscale_it  # We got rid of the non_it_power()
            Rscale_it = N * (P_on_avg - simulator._idle_power) / 2
            Rscale = Rscale_it
        print('\nPratio, Rratio: %f, %f' % (P / Pscale, R / Rscale))
        alpha, delay_prob = self.calculate_delay_prob(fit=True, sim_hour=sim_hour, J=J,
                                                      job_table=job_table, QoS_constraint=QoS_constraint,
                                                      min_execution_time=min_execution_time,
                                                      dc_index=dc_index)
        print('Degradation probability: %s%%' % (str([int(100 * x) for x in delay_prob])))

        # Read current values from results_list
        c_k = results_list[dc_index]['c_k']
        c_t = results_list[dc_index]['c_t']
        mu_t = results_list[dc_index]['mu_t']
        mu_j_values = results_list[dc_index]['mu_j_values']
        # Read rates from results_list if available, otherwise use from self
        c_k_rate = results_list[dc_index].get('c_k_rate', self.c_k_rate_list[dc_index])
        c_t_rate = results_list[dc_index].get('c_t_rate', self.c_t_rate_list[dc_index])

        dCdP, dCdR, dCdw, dCd_qos_lagrange, dCd_tracking_lagrange, cost_power, cost_error, cost_qos, cost_standby, qos_violation, tracking_violation, new_mu_j_values, new_mu_t = \
            self.calculate_gradient(iteration=iteration,
                                    beta=beta, rho=rho,
                                    sim_hour=sim_hour,
                                    simulator=simulator,
                                    alpha=alpha,
                                    delay_prob=delay_prob,
                                    job_power_max=max_job_power,
                                    job_size=job_size,
                                    dc_index=dc_index,
                                    c_k=c_k,
                                    c_t=c_t,
                                    mu_t=mu_t,
                                    mu_j_values=mu_j_values)

        cost_theory = cost_power + cost_error + cost_qos + cost_standby
        print('Cost_theory: %.3f=%.3f+%.3f+%.3f+%.3f' % (cost_theory, cost_power, cost_error, cost_qos, cost_standby))
        if iteration == 0:
            with open(self.output_dir + 'GD' + '_dc' + str(dc_index) + '.csv', 'w') as out:
                out.write('iter,cost_theory,costPower,cost_error,cost_QoS,P,R,Pcoef,Rcoef,' + ','.join(
                    ['weight%d' % x for x in range(1, J + 1)]) + ',' + ','.join(
                    ['prob%d' % x for x in range(1, J + 1)]) + '\n')
        with open(self.output_dir + 'GD' + '_dc' + str(dc_index) + '.csv', 'a') as out:
            out.write('%d,%f,%f,%f,%f,%.f,%.f,%f,%f,' % (
                iteration, cost_theory, cost_power, cost_error, cost_qos, P, R, P / Pscale, R / Rscale) + ','.join(
                [str(x) for x in w]) + ',' + ','.join([str(x) for x in delay_prob]) + '\n')

        gradients = {
            'dCdP': dCdP,
            'dCdR': dCdR,
            'dCdw': dCdw,
            # 'dCd_qos_lagrange': dCd_qos_lagrange,
            # 'dCd_tracking_lagrange': dCd_tracking_lagrange,
            'w': w,
            'J': J,
            'P': P,
            'R': R,
            'Pscale': Pscale,
            'Rscale': Rscale,
            'cost_theory': cost_theory,
            'cost_power': cost_power,
            'cost_error': cost_error,
            'cost_qos': cost_qos,
            'job_table_length': job_table.shape[0]
        }
        gradients = [gradients]
        # Calculate the batch gradient
        dCdP = sum(g['dCdP'] for g in gradients) / len(gradients)
        dCdR = sum(g['dCdR'] for g in gradients) / len(gradients)
        dCdw = np.sum([g['dCdw'] for g in gradients], axis=0) / len(gradients)
        # dCd_qos_lagrange = sum(g['dCd_qos_lagrange'] for g in gradients) / len(gradients)
        # dCd_tracking_lagrange = sum(g['dCd_tracking_lagrange'] for g in gradients) / len(gradients)

        w = gradients[0]['w']
        J = gradients[0]['J']
        Pscale = gradients[0]['Pscale']
        Rscale = gradients[0]['Rscale']
        P = gradients[0]['P']
        R = gradients[0]['R']

        # Update c_k and c_t based on violations (similar to qos_lagrange pattern)
        new_c_k = c_k
        new_c_t = c_t

        if self.use_lagrange:
            # Update augmented Lagrangian penalty parameters using configurable rates
            if qos_violation:
                new_c_k = c_k * c_k_rate
            if tracking_violation:
                new_c_t = c_t * c_t_rate

        weights_dict = {job_type: weight for job_type, weight in zip(self.job_configs[dc_index].all_jobs.values(), w)}
        wandb_dict = {}
        if self.logger == 'wandb':
            mu_j_vals_dict = {}
            for index, mu_j in enumerate(new_mu_j_values):
                mu_j_vals_dict[f'mu_j_{index}'] = mu_j

            wandb_dict = {
                f"{key}_dc{dc_index}": value for key, value in {
                    "P": P,
                    "R": R,
                    "cost_theory": gradients[0]['cost_theory'],
                    "cost_power": gradients[0]['cost_power'],
                    "cost_error": gradients[0]['cost_error'],
                    "cost_qos": gradients[0]['cost_qos'],
                    "weights": weights_dict,
                    "job_table_length": gradients[0]['job_table_length'],
                    # "qos_lagrange": results_list[dc_index]['qos_lagrange'],
                    # "dCd_qos_lagrange": dCd_qos_lagrange,
                    # "tracking_lagrange": results_list[dc_index]['tracking_lagrange'],
                    # "dCd_tracking_lagrange": dCd_tracking_lagrange,
                    "ratio_of_job_type_qos_violation": simulator.ratio_of_job_type_qos_violation,
                    "tracking_error_at_90": simulator.tracking_error_at_90,
                    "mu_j_values": mu_j_vals_dict,
                    "mu_t": new_mu_t,
                    "c_k": new_c_k,
                    "c_t": new_c_t,
                    "c_k_rate": c_k_rate,
                    "c_t_rate": c_t_rate
                }.items()
            }

            # wandb.log({"P": P, "R": R, "cost_theory": gradients[0]['cost_theory'],
            #            "cost_power": gradients[0]['cost_power'], "cost_error": gradients[0]['cost_error'],
            #            "cost_qos": gradients[0]['cost_qos'], "weights": weights_dict,
            #            "job_table_length": gradients[0]['job_table_length'],
            #            "qos_lagrange": self.qos_lagrange,
            #            "dCd_qos_lagrange": dCd_qos_lagrange,
            #            "tracking_lagrange": self.tracking_lagrange,
            #            "dCd_tracking_lagrange": dCd_tracking_lagrange,
            #            "ratio_of_job_type_qos_violation": simulator.ratio_of_job_type_qos_violation,
            #            "tracking_error_at_90": simulator.tracking_error_at_90})

        # tune dw.
        dw = [-kappa * dCdw[itype] for itype in range(J)]
        s = sum(dw)
        dw = [dw[i] - s / J for i in range(J)]
        while any(abs(dw[x]) > w_step_up * np.exp(-iteration / limit_decrease) for x in
                  range(J)):  # w update too much.
            kappa = kappa / 2
            dw = [dw[i] / 2 for i in range(J)]
        while all(abs(dw[x]) < w_step_down * np.exp(-iteration / limit_decrease) for x in
                  range(J)):  # w update too little.
            if sum([abs(i) for i in dw]) == 0:
                break
            kappa = kappa * 2
            dw = [dw[i] * 2 for i in range(J)]
        print('kappa: %s' % str(kappa))

        # tune dP, dR.
        dP, dR = -kappa * dCdP * phi, -kappa * dCdR * phi
        print(f'Initial dP: {dP}, initial dR: {dR}, dCdP: {dCdP}, dCdR: {dCdR}')
        while abs(dP) > PR_up * np.exp(-iteration / limit_decrease) or abs(dR) > PR_up * np.exp(
                -iteration / limit_decrease):
            dP, dR = dP / 2, dR / 2
        while (abs(dP) < PR_down * np.exp(-iteration / limit_decrease) and
               abs(dR) < PR_down * np.exp(-iteration / limit_decrease)):
            dP, dR = dP * 2, dR * 2
        print('dP: %.1f, dR: %.1f, from P: %.1f, R: %.1f' % (dP, dR, P, R))

        # tune dw again. only update those w whose w+dw is not too small.
        positive = []
        for itype in range(J):
            if w[itype] + dw[itype] >= wmin:
                positive.append(itype)
        s = sum([dw[i] for i in positive])
        dw = [dw[i] - s / len(positive) if i in positive else 0 for i in range(J)]
        while all(abs(dw[x]) < w_step_down * np.exp(-iteration / limit_decrease) for x in
                  positive):  # w update too little.
            if sum([abs(i) for i in dw]) == 0:
                break
            dw = [dw[i] * 2 for i in range(J)]
        print("dw:", dw)

        for itype in positive:
            w[itype] = w[itype] + dw[itype]

        for idx, x in enumerate(w):
            if x < 0:
                offset = -x
                w[idx] = 0
                for jdx, y in enumerate(w):
                    if idx != jdx:
                        w[jdx] -= offset / (len(w) - 1)
        # weights = w

        P, R = P + dP, R + dR
        weights = [round(x, 6) for x in w]  # This mimics the weights precision of old optimizer
        print("weights:", weights)

        result_dict = {'P': P, 'R': R, 'Pscale': Pscale, 'Rscale': Rscale, 'weights': weights,
                       'weighted_average_power': simulator._runtime_policy.weighted_average_power,
                       'wandb': wandb_dict,
                       'c_k': new_c_k,
                       'c_t': new_c_t,
                       'c_k_rate': c_k_rate,
                       'c_t_rate': c_t_rate,
                       'mu_t': new_mu_t,
                       'mu_j_values': new_mu_j_values
                       }

        results_list[dc_index] = result_dict

    def gradient_driver(self):
        inits_list = []
        for dc_index in range(self.num_data_centers):
            (initP, initR) = (self.gradient_configs[dc_index].general_variables.values())

            p_scale, r_scale = 0, 0

            weights = 0
            job_ids = np.array(list(self.job_configs[dc_index].all_jobs.values()))
            base_weights_path = os.path.join(self.output_dir, 'base_weights' + '_dc' + str(dc_index) + '.csv')
            pd.DataFrame({
                'job_type_id': job_ids,
                'weights': 1.0 / (job_ids.size or 1.0)
            }).to_csv(base_weights_path, index=False)
            self.policy_configs[dc_index].policy_parameters['weights_path'] = base_weights_path

            P, R = denormalize_p_and_r_to_watts(self.experiment_configs[dc_index], self.job_configs[dc_index],
                                                initP,
                                                initR)
            self.policy_configs[dc_index].p_ratio = initP
            self.policy_configs[dc_index]._P = P
            self.policy_configs[dc_index]._R = R

            inits_list.append({"P": initP, "R": initR, "Pscale": p_scale, "Rscale": r_scale,
                               "weights": weights,
                               # "qos_lagrange": self.qos_lagrange_list[dc_index],
                               # "tracking_lagrange": self.tracking_lagrange_list[dc_index],
                               "c_k": self.c_k_list[dc_index],
                               "c_t": self.c_t_list[dc_index],
                               "c_k_rate": self.c_k_rate_list[dc_index],
                               "c_t_rate": self.c_t_rate_list[dc_index],
                               "mu_t": self.mu_t_list[dc_index],
                               "mu_j_values": self.mu_j_values_list[dc_index],
                               "weighted_average_power": None})

        with multiprocessing.Manager() as manager:
            results_list = manager.list(inits_list)
            for iteration in range(int(self.num_iterations)):
                print("\n\n======================", "Iteration:", iteration, "======================\n")

                qos_values = manager.dict()
                power_assignments = manager.dict()
                stop_signal = manager.Value('b', False)
                current_time_shared = manager.Value('i', 0)
                # assignment_conditions = [manager.Condition() for _ in range(self.num_data_centers)]
                assignment_events = [manager.Event() for _ in range(self.num_data_centers)]
                power_balancer_condition = manager.Condition()
                qos_write_lock = manager.Lock()

                # Launch power balancer
                power_balancer = PowerBalancer(self.num_data_centers, self.policy_configs,
                                               self.experiment_configs, self.job_configs, stop_signal,
                                               assignment_events, power_balancer_condition,
                                               qos_write_lock)

                power_balancer_process = multiprocessing.Process(
                    target=power_balancer.run,
                    args=(current_time_shared, qos_values, power_assignments)
                )
                power_balancer_process.start()

                # Launch data center sim processes
                dc_processes = []

                for dc_index in range(self.num_data_centers):
                    p = multiprocessing.Process(
                        target=self.increment_gradient,
                        args=(iteration,
                              results_list[dc_index]['P'],
                              results_list[dc_index]['R'],
                              results_list[dc_index]['Pscale'],
                              results_list[dc_index]['Rscale'],
                              results_list[dc_index]['weights'],
                              results_list[dc_index]['weighted_average_power'],
                              dc_index,
                              qos_values,
                              power_assignments,
                              current_time_shared,
                              results_list,
                              assignment_events[dc_index],
                              power_balancer_condition,
                              qos_write_lock
                              )
                    )
                    p.start()
                    dc_processes.append(p)

                for dc_index, p in enumerate(dc_processes):
                    p.join()
                    self.policy_configs[dc_index]._P = results_list[dc_index]['P']
                    self.policy_configs[dc_index]._R = results_list[dc_index]['R']

                stop_signal.value = True
                with power_balancer_condition:
                    power_balancer_condition.notify_all()
                power_balancer_process.join()

                if self.logger == 'wandb':
                    merged_dict = {}
                    for dc_index in range(len(results_list)):
                        merged_dict.update(results_list[dc_index]['wandb'])
                    wandb.log(merged_dict)
            for dc_index in range(self.num_data_centers):
                weights_df = pd.DataFrame({
                    'job_type_id': list(self.job_configs[dc_index].all_jobs.values()),
                    'weights': results_list[dc_index]["weights"]
                })

                print('\n\n============> GD finished \n Writing weights to ' + self.output_dir
                      + 'opt_weights_dc' + str(dc_index) + '.csv')

                weights_df.to_csv(self.output_dir + 'opt_weights_dc' + str(dc_index) + '.csv', index=False)

                # Create a ConfigParser object
                print('Writing test policy configuration ')

                df = pd.read_csv(self.output_dir + "GD_dc" + str(dc_index) + ".csv")

                # Extract the last row
                last_row = df.iloc[-1]

                # Extract required variables
                P = last_row['P']
                R = last_row['R']
                P_ratio = last_row['Pcoef']
                R_ratio = last_row['Rcoef']

                policy_name = self.policy_configs[dc_index].runtime_policy_name
                weights_path = self.output_dir + "opt_weights_dc" + str(dc_index) + ".csv"

                config = configparser.ConfigParser()

                config['bid'] = {
                    'P': str(P),
                    'R': str(R),
                    'P_ratio': str(P_ratio),
                    'R_ratio': str(R_ratio)
                }

                config['policy'] = {
                    'name': policy_name
                }

                config['parameters'] = {
                    'weights_path': weights_path
                }

                with open(self.output_dir + 'test_policy_dc' + str(dc_index) + '.ini', 'w') as configfile:
                    config.write(configfile)

    def run(self):
        np.random.seed(self.experiment_configs[0].random_seed)
        random.seed(self.experiment_configs[0].random_seed)
        self.gradient_driver()

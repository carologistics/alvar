/*
 * This file is part of ALVAR, A Library for Virtual and Augmented Reality.
 *
 * Copyright 2007-2012 VTT Technical Research Centre of Finland
 *
 * Contact: VTT Augmented Reality Team <alvar.info@vtt.fi>
 *          <http://www.vtt.fi/multimedia/alvar.html>
 *
 * ALVAR is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ALVAR; if not, see
 * <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.
 */

#include "UnscentedKalman.h"

#include <opencv2/core.hpp>
#include <stdio.h>

namespace alvar {

UnscentedKalman::UnscentedKalman(int state_n, int obs_n, int state_k, double alpha, double beta)
{
	state_k = 0;
	//TODO: support a separate noise vector/covariance matrix: state_k;
	this->state_k = state_k;
	this->state_n = state_n;
	this->obs_n   = obs_n;
	sigma_n       = 2 * state_n + 1;

	double L = state_n + state_k;
	lambda   = alpha * alpha * L - L;
	lambda2  = 1 - alpha * alpha + beta;

	state = cv::Mat::zeros(state_n, 1, CV_64F);

	stateCovariance     = cv::Mat::zeros(state_n, state_n, CV_64F);
	sqrtStateCovariance = cv::Mat::zeros(state_n, state_n, CV_64F);
	stateD              = cv::Mat::zeros(state_n, state_n, CV_64F);
	stateU              = cv::Mat::zeros(state_n, state_n, CV_64F);
	stateV              = cv::Mat::zeros(state_n, state_n, CV_64F);
	stateTmp            = cv::Mat::zeros(state_n, state_n, CV_64F);
	stateDiff           = cv::Mat::zeros(state_n, 1, CV_64F);

	predObs           = cv::Mat::zeros(obs_n, 1, CV_64F);
	predObsCovariance = cv::Mat::zeros(obs_n, obs_n, CV_64F);
	predObsDiff       = cv::Mat::zeros(obs_n, 1, CV_64F);

	invPredObsCovariance         = cv::Mat::zeros(obs_n, obs_n, CV_64F);
	statePredObsCrossCorrelation = cv::Mat::zeros(state_n, obs_n, CV_64F);
	kalmanGain                   = cv::Mat::zeros(state_n, obs_n, CV_64F);
	kalmanTmp                    = cv::Mat::zeros(state_n, obs_n, CV_64F);

	sigma_state   = std::vector<cv::Mat>(sigma_n, cv::Mat());
	sigma_predObs = std::vector<cv::Mat>(sigma_n, cv::Mat());

	for (int i = 0; i < sigma_n; i++) {
		sigma_state[i]   = cv::Mat::zeros(state_n, 1, CV_64F);
		sigma_predObs[i] = cv::Mat::zeros(obs_n, 1, CV_64F);
	}

	sigmasUpdated = false;
}

UnscentedKalman::~UnscentedKalman()
{
	state.release();
	stateCovariance.release();
	sqrtStateCovariance.release();
	stateD.release();
	stateU.release();
	stateV.release();
	stateTmp.release();
	stateDiff.release();
	kalmanTmp.release();
	kalmanGain.release();
	statePredObsCrossCorrelation.release();
	invPredObsCovariance.release();
	predObs.release();
	predObsCovariance.release();
	predObsDiff.release();

	for (int i = 0; i < sigma_n; i++) {
		sigma_state[i].release();
		sigma_predObs[i].release();
	}
}

void
UnscentedKalman::initialize()
{
	// Computes new sigma points from current state estimate.

	// 1. Compute square root of state co-variance:
	//[E D] = eig(A); sqrtm(A) = E * sqrt(D) * E' where D is a diagonal matrix.
	//sqrt(D) is formed by taking the square root of the diagonal entries in D.
#ifdef MYDEBUG
	printf("stateCovariance:\n");
	for (int i = 0; i < 5; i++)
		printf("%+.10f %+.10f %+.10f %+.10f %+.10f\n",
               stateCovariance.at<double>(0, i),
               stateCovariance.at<double>(1, i),
               stateCovariance.at<double>(2, i),
               stateCovariance.at<double>(3, i),
               stateCovariance.at<double>(4, i),
#endif

	//Another equivilant way is to use:
	// [U S V] = svd(A); sqrtm(A) = U * sqrt(S) * V'
    cv::SVD::compute(stateCovariance, stateD, stateU, stateV); //, CV_SVD_V_T
	double L     = state_n + state_k;
	double scale = L + lambda;
	for (int i = 0; i < state_n; i++) {
			double d                = stateD.at<double>(i, i);
			stateD.at<double>(i, i) = sqrt(scale * d);
	}
	cv::gemm(stateD, stateV, 1., NULL, 0, stateTmp, cv::GEMM_2_T);
	cv::gemm(stateU, stateTmp, 1., NULL, 0, sqrtStateCovariance);
#ifdef MYDEBUG
	printf("sqrtStateCovariance:\n");
	for (int i = 0; i < 5; i++)
		printf("%+.10f %+.10f %+.10f %+.10f %+.10f\n",
		       sqrtStateCovariance.at<double>(0, 1),
		       sqrtStateCovariance.at<double>(1, 1),
		       sqrtStateCovariance.at<double>(2, 1),
		       sqrtStateCovariance.at<double>(3, 1),
		       sqrtStateCovariance.at<double>(4, 1));
	cv::gemm(sqrtStateCovariance, sqrtStateCovariance, 1., NULL, 0, stateTmp);
	printf("sqrtStateCovariance^2:\n");
	for (int i = 0; i < 5; i++)
		printf("%+.10f %+.10f %+.10f %+.10f %+.10f\n",
		       stateTmp.at<double>(0, i),
		       stateTmp.at<double>(1, i),
		       stateTmp.at<double>(2, i),
		       stateTmp.at<double>(3, i),
		       stateTmp.at<double>(4, i));
#endif

	// 2. Form new sigma points.
	int sigma_i = 0;
    state.copyTo(sigma_state[sigma_i++]);
	for (int i = 0; i < state_n; i++) {
			cv::Mat col;
			col                    = sqrtStateCovariance.col(i);
			sigma_state[sigma_i++] = state + col;
			sigma_state[sigma_i++] = state - col;
	}

	sigmasUpdated = true;
}

void
UnscentedKalman::predict(UnscentedProcess *process_model)
{
	if (!sigmasUpdated)
		initialize();

	// Map sigma points through the process model and compute new state mean.
	state              = cv::Mat::zeros(state.size(), state.type());
	double L           = state_n + state_k;
	double totalWeight = 0;
	for (int i = 0; i < sigma_n; i++) {
		double weight = i == 0 ? lambda / (L + lambda) : .5 / (L + lambda);
		totalWeight += weight;
	}
	for (int i = 0; i < sigma_n; i++) {
		cv::Mat sigma = sigma_state[i];
		process_model->f(sigma);
		double weight = i == 0 ? lambda / (L + lambda) : .5 / (L + lambda);
		double scale  = weight / totalWeight;
		cv::addWeighted(sigma, scale, state, 1., 0., state);
	}

	// Compute new state co-variance.
	stateCovariance = cv::Mat::zeros(stateCovariance.size(), stateCovariance.type());
	totalWeight     = 0;
	for (int i = 0; i < sigma_n; i++) {
		double weight = i == 0 ? lambda / (L + lambda) + lambda2 : .5 / (L + lambda);
		totalWeight += weight;
	}
	for (int i = 0; i < sigma_n; i++) {
		double weight = i == 0 ? lambda / (L + lambda) + lambda2 : .5 / (L + lambda);
		double scale  = weight / totalWeight;
		stateDiff     = sigma_state[i] - state;
		cv::gemm(stateDiff, stateDiff, scale, stateCovariance, 1., stateCovariance, cv::GEMM_2_T);
	}

	// Add any additive noise.
	cv::Mat noise = process_model->getProcessNoise();
	if (!noise.empty())
		stateCovariance = noise + stateCovariance;

#ifdef MYDEBUG
	printf("predicted state: ");
	for (int i = 0; i < state_n; i++)
		printf("%f ", state.at<double>(i));
	printf("\n");
	printf("predicted stateCovariance:\n");
	for (int i = 0; i < state_n; i++) {
		for (int j = 0; j < state_n; j++)
			printf("%+f ", stateCovariance.at<double>(i, j));
		printf("\n");
	}
#endif

	sigmasUpdated = false;
}

void
UnscentedKalman::update(UnscentedObservation *obs)
{
	if (!sigmasUpdated)
		initialize();
	cv::Mat innovation = obs->getObservation();
	int     obs_n      = innovation.rows;
	if (obs_n > this->obs_n) {
		printf("Observation exceeds maximum size!\n");
		abort();
	}

	// Map sigma points through the observation model and compute predicted mean.
	cv::Mat predObs = cv::Mat::zeros(obs_n, 1, CV_64F);
	for (int i = 0; i < sigma_n; i++) {
		cv::Mat sigma_h = cv::Mat(obs_n, 1, CV_64F, sigma_predObs[i].data);
		double  scale =
      i == 0 ? (double)state_k / (double)(state_n + state_k) : .5 / (double)(state_n + state_k);
		obs->h(sigma_h, sigma_state[i]);
		cv::addWeighted(sigma_h, scale, predObs, 1., 0., predObs);
	}

	// Compute predicted observation co-variance.
	cv::Mat predObsCovariance = cv::Mat(obs_n, obs_n, CV_64F, this->predObsCovariance.data);
	cv::Mat statePredObsCrossCorrelation =
	  cv::Mat(state_n, obs_n, CV_64F, this->statePredObsCrossCorrelation.data);
	cv::Mat predObsDiff = cv::Mat(obs_n, 1, CV_64F, this->predObsDiff.data);
	predObsCovariance   = cv::Mat::zeros(predObsCovariance.size(), predObsCovariance.type());
	statePredObsCrossCorrelation =
	  cv::Mat::zeros(statePredObsCrossCorrelation.size(), statePredObsCrossCorrelation.type());
	for (int i = 0; i < sigma_n; i++) {
		cv::Mat sigma_h = cv::Mat(obs_n, 1, CV_64F, sigma_predObs[i].data);
		double  scale =
      i == 0 ? (double)state_k / (double)(state_n + state_k) : .5 / (double)(state_n + state_k);
		stateDiff   = sigma_state[i] - state;
		predObsDiff = sigma_h - predObs;
		cv::gemm(
		  predObsDiff, predObsDiff, scale, predObsCovariance, 1., predObsCovariance, cv::GEMM_2_T);
		cv::gemm(stateDiff,
		         predObsDiff,
		         scale,
		         statePredObsCrossCorrelation,
		         1.,
		         statePredObsCrossCorrelation,
		         cv::GEMM_2_T);
	}

	// Add any additive noise.
	cv::Mat noise = obs->getObservationNoise();
	if (!noise.empty())
		predObsCovariance = predObsCovariance + noise;

#ifdef MYDEBUG
	printf("real observation: ");
	for (int i = 0; i < obs_n; i++)
		printf("%+f ", cvGetReal1D(innovation, i));
	printf("\n");
	printf("predicted observation: ");
	for (int i = 0; i < obs_n; i++)
		printf("%+f ", cvGetReal1D(&predObs, i));
	printf("\n");
	printf("predicted observation co-variance\n");
	for (int i = 0; i < obs_n; i++) {
		for (int j = 0; j < obs_n; j++)
			printf("%+f ", predObsCovariance.at<double>(i, j));
		printf("\n");
	}
	printf("state observation cross-correlation\n");
	for (int i = 0; i < state_n; i++) {
		for (int j = 0; j < obs_n; j++)
			printf("%+f ", statePredObsCrossCorrelation.at<double>(i, j));
		printf("\n");
	}
#endif

	// Update state mean and co-variance.
	//  innovation: v = z - pz
	//  gain: W = XZ * (R + Z)^-1
	//  state: x = x + _W * v
	//  co-var: P = P - W * (R + Z) * W^T

	cv::Mat invPredObsCovariance = cv::Mat(obs_n, obs_n, CV_64F, this->invPredObsCovariance.data);
	cv::Mat kalmanGain           = cv::Mat(state_n, obs_n, CV_64F, this->kalmanGain.data);
	cv::Mat kalmanTmp            = cv::Mat(state_n, obs_n, CV_64F, this->kalmanTmp.data);
	innovation                   = innovation - predObs;
	//double inno_norm = cvNorm(innovation) / obs_n;
	//if (inno_norm > 5.0) {
	//  return;
	//}

#ifdef MYDEBUG
	printf("innovation: ");
	for (int i = 0; i < obs_n; i++)
		printf("%f ", cvGetReal1D(innovation, i));
	printf("\n");
	double inn_norm = cv::norm(innovation);
	printf("innivation norm: %f\n", inn_norm);
#endif

	cv::invert(predObsCovariance, invPredObsCovariance, CV_HAL_SVD_SHORT_UV);
	kalmanGain = statePredObsCrossCorrelation * invPredObsCovariance;
	cv::gemm(kalmanGain, innovation, 1., state, 1., state);
	kalmanTmp = kalmanGain * predObsCovariance;
	cv::gemm(kalmanTmp, kalmanGain, -1., stateCovariance, 1., stateCovariance, cv::GEMM_2_T);
#ifdef MYDEBUG
	printf("estimated state: ");
	for (int i = 0; i < state_n; i++)
		printf("%f ", cvGetReal1D(state, i));
	printf("\n");
#endif

	sigmasUpdated = false;
}

} // namespace alvar

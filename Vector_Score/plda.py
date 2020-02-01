#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plda.py
@Time: 2019/12/6 上午10:42
@Overview: MOdified from kaldi. The original script is 'kaldi/src/ivector/plda.h'
"""
import numpy as np

M_LOG_2PI = 1.8378770664093454835606594728112

def ApplyFloor(np, floor_val):
    n = 0

    for i in range(len(np)):
        if np[i] < floor_val:
            np[i] = floor_val
            n +=1
    return n

def Resize(shape):
    return np.zeros(shape=shape)

class PldaConfig:
    """
    normalize_length: "If true, do length normalization as part of PLDA (see "
                   "code for details).  This does not set the length unit; "
                   "by default it instead ensures that the inner product "
                   "with the PLDA model's inverse variance (which is a "
                   "function of how many utterances the iVector was averaged "
                   "over) has the expected value, equal to the iVector "
                   "dimension."
    simple_length_norm: "If true, replace the default length normalization by an "
                   "alternative that normalizes the length of the iVectors to "
                   "be equal to the square root of the iVector dimension.
    """

    def __init__(self):
        self.normalize_length = True
        self.simple_length_norm = False

    def register(self, **kwargs):
        if 'normalize_length' in  kwargs.keys():
            self.normalize_length = kwargs['normalize_length']
        if 'simple_length_norm' in kwargs.keys():
            self.simple_length_norm = kwargs['simple_length_norm']

class PLDA:

    def __init__(self):

        self.mean_
        self.transform_
        self.psi_
        self.offset_

    def Dim(self):
        return len(self.mean_)

    def ComputeNormalizingTransform(self, covar):

        C = np.linalg.cholesky(covar)
        C = 1 / C
        return C

    def ComputeDerivedVars(self):

        assert (self.Dim() > 0)
        # self.offset_.re(Dim())
        self.offset_ = -1.0 * self.transform_ * self.mean_

    def GetNormalizationFactor(self, transformed_ivector, num_examples):
        assert (num_examples > 0)
        #  Work out the normalization factor. The covariance for an average over "num_examples"
        #  training iVectors equals \Psi + I / num_examples.
        transformed_ivector_sq = transformed_ivector
        transformed_ivector_sq = np.square(transformed_ivector_sq)
        #  inv_covar will equal 1.0 / (\Psi + I / num_examples).
        inv_covar = self.psi_
        inv_covar += 1.0 / num_examples

        inv_covar = 1./ inv_covar
        # "transformed_ivector" should have covariance(\Psi + I / num_examples), i.e.
        # within-class /num_examples plus between- class covariance.So
        # transformed_ivector_sq.(I / num_examples + \Psi) ^ {-1} should be equal to
        # the dimension.
        dot_prod = np.matmul(inv_covar, transformed_ivector_sq)

        return np.sqrt(self.Dim / dot_prod)

    def TransformIvector(self, config, ivector, num_examples, transformed_ivector):

        assert (len(ivector) == self.Dim() and transformed_ivector.Dim == self.Dim())
        transformed_ivector = self.offset_
        transformed_ivector = self.transform_ * transformed_ivector * ivector

        if (config.simple_length_norm):
            normalization_factor = np.sqrt(transformed_ivector.Dim) / np.sqrt(np.square(transformed_ivector))
        else:
            normalization_factor = self.GetNormalizationFactor(transformed_ivector, num_examples);

        if (config.normalize_length):
            transformed_ivector = transformed_ivector * normalization_factor

        return normalization_factor

    def LogLikelihoodRatio(self,transformed_train_ivector, n, #  number of training utterances.
                           transformed_test_ivector):
        dim = self.Dim
        # loglike_given_class, loglike_without_class
        # {
        # work out loglike_given_class. "mean" will be the mean of the distribution if it comes from the
        # training example. The mean is \frac {n \Psi}{n \Psi + I} \bar {u} ^ g.
        # "variance" will be the variance of that distribution, equal to I + \frac{\Psi}{n\Psi + I}.
        mean = np.array(dim)
        variance = np.array(dim)

        for i in range(0, dim):
            mean[i] = n * self.psi_[i]/ (n * self.psi_[i] + 1.0) * transformed_train_ivector(i)
            variance[1] = 1.0 + self.psi_[i] / (n * self.psi_[i] + 1.0)

        logdet = np.log(np.sum(variance))
        sqdiff = transformed_test_ivector
        sqdiff -= mean
        sqdiff = np.square(sqdiff)
        variance = 1/ variance

        loglike_given_class = -0.5 * (logdet + M_LOG_2PI * dim + sqdiff * variance)
        # }
        # {// work out loglike_without_class.Here the mean is zero and the variance is I + \Psi.
        sqdiff = transformed_test_ivector #  there is no offset.
        sqdiff = np.square(sqdiff)

        variance = self.psi_
        variance = 1 + variance # I + \Psi.
        logdet = np.log(np.sum(variance))
        variance = 1 / variance
        loglike_without_class = -0.5 * (logdet + M_LOG_2PI * dim + sqdiff * variance)
        # }
        loglike_ratio = loglike_given_class - loglike_without_class;

        return loglike_ratio;

    def SmoothWithinClassCovariance(self, smoothing_factor):

        assert (smoothing_factor >= 0.0 and smoothing_factor <= 1.0)
        # smoothing_factor > 1.0 is possible but wouldn't really make sense.
        print("Smoothing within-class covariance by " + str(smoothing_factor) + ", Psi is initially: " + str(self.psi_))
        within_class_covar = self.Dim
        within_class_covar.Set(1.0); # It's now the current within-class covariance
                                     # (a diagonal matrix) in the space transformed
                                     # by transform_.
        within_class_covar += smoothing_factor * self.psi_
        # We now revise our estimate of the within-class covariance to this
        # larger value.  This means that the transform has to change to as
        # to make this new, larger covariance unit.  And our between-class
        # covariance in this space is now less.

        self.psi_ /= within_class_covar
        print("New value of Psi is " + str(self.psi_))

        within_class_covar.ApplyPow(-0.5);
        self.transform_.MulRowsVec(within_class_covar);

        self.ComputeDerivedVars()

    def ApplyTransform(self, in_transform):
        assert (len(in_transform)<=self.Dim and in_transform.shape[1]==self.Dim)

        # Apply in_transform to mean_.
        mean_new = in_transform * self.mean_
        self.mean_ = mean_new

        transform_invert = self.transform_

        # Next, compute the between_var and within_var that existed
        # prior to diagonalization.
        psi_mat = np.diag(np.diag(self.psi_))
        transform_invert = 1 / transform_invert

        within_var = transform_invert
        between_var = transform_invert * psi_mat

        # Next, transform the variances using the input transformation.
        between_var_new = in_transform * between_var
        within_var_new = in_transform * within_var

        # Finally, we need to recompute psi_ and transform_. The remainder of
        # the code in this function  is a lightly modified copy of
        # PldaEstimator::GetOutput().
        transform1 = self.ComputeNormalizingTransform(within_var_new)

        # Now transform is a matrix that if we project with it, within_var becomes unit.
        # between_var_proj is between_var after projecting with transform1.
        between_var_proj = transform1 * between_var_new

        # Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
        # where U is orthogonal.
        s, U = np.linalg.eig(between_var_proj)

        assert (s.min() >=0 )
        n = ApplyFloor(s, 0.0)

        if n>0:
            print("Floored " + str(n) + " eigenvalues of between-class variance to zero.")

        # Sort from greatest to smallest eigenvalue.
        sortindex_s = np.argsort(-s)
        s = s[sortindex_s]
        U = U[sortindex_s]

        # The transform U^T will make between_var_proj diagonal with value s
        # (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
        # makes within_var unit and between_var diagonal is U^T transform1,
        # i.e. first transform1 and then U^T.
        self.transform_ = np.transpose(U) * transform1
        self.psi_ = s

        self.ComputeDerivedVars()

class ClassInfo:
    def __init__(self):
        self.weight
        self.mean
        self.num_examples

class PldaStats:

    def __init__(self, dim):
        self.dim_ = dim
        self.num_classes_ = 0
        self.num_examples_ = 0        # total number of examples, summed over classes.
        self.class_weight_ = 0.0      # total over classes, of their weight.
        self.example_weight_ = 0.0  # total over classes, of weight times #examples.
        self.sum_ = np.zeros(dim)
                             # Weighted sum of class means (normalize by class_weight_ to get mean).
                             # class means(normalize by
                             # class_weight_ to get mean).
                             # 使用weight计算的所有类的均值
        self.offset_scatter_ = np.zeros((dim, dim))
                             # Sum over all examples, of the weight
                             # times (example - class-mean).
                             # 所有egs的weight加权均值

        self.class_info_ = ClassInfo()


    def Dim(self):
        return self.dim_

    def AddSamples(self, weight, group):
        assert (self.dim_==len(group))
        n = len(group)
        mean = np.zeros(self.dim_)
        mean += group.sum(axis=0) / n

        #   mean->AddRowSumMat(1.0 / n, group);
        #
        self.offset_scatter_ += weight * (group.transpose() * group)

        #   // the following statement has the same effect as if we
        #   // had first subtracted the mean from each element of
        #   // the group before the statement above.
        self.offset_scatter_ += -n * weight * np.square(mean)
        #
        #   class_info_.push_back(ClassInfo(weight, mean, n));
        #
        self.num_classes_ += 1
        self.num_examples_ += n
        self.class_weight_ += weight
        self.example_weight_ += weight * n
        self.sum_ += weight * mean

class PldaEstiator:
    def __init__(self):
        self.class_info_
        self.stats_

        # 类内方差、类间方差
        self.within_var_
        self.between_var_
        # These stats are reset on each iteration.
        # 类内方差统计量
        self.within_var_stats_

        # 计算类内方差的样本数
        self.within_var_count_ # count corresponding to within_var_stats_

        # 类间方差统计量
        self.between_var_stats_

        # 计算类间方差的样本数
        self.between_var_count_ # count corresponding to within_var_stats_

        #   KALDI_DISALLOW_COPY_AND_ASSIGN(PldaEstimator);
    def InitParameters(self):
        self.within_var_ = Resize(self.Dim())
        self.within_var_ = np.identity(len(self.within_var_))
        self.between_var_ = Resize(self.Dim())
        self.between_var_ = np.identity(len(self.between_var_))

    def ComputeObjfPart1(self):
        # Returns the part of the obj relating to the class means (total_not normalized)
        # 计算类均值
        # double within_class_count = stats_.example_weight_ - stats_.class_weight_,
        #       within_logdet, det_sign;
        within_class_count = self.stats_.example_weight_ - self.stats_.class_weight_
        within_logdet=0
        det_sign=0

        #   SpMatrix<double> inv_within_var(within_var_);
        #   inv_within_var.Invert(&within_logdet, &det_sign);
        #   KALDI_ASSERT(det_sign == 1 && "Within-class covariance is singular");
        #
        #   double objf = -0.5 * (within_class_count * (within_logdet + M_LOG_2PI * Dim())
        #                         + TraceSpSp(inv_within_var, stats_.offset_scatter_));
        #   return objf;

    # def ComputeObjfPart2(self):

        # Returns the objective-function per sample.
        # 计算每个egs的目标函数？

    # def ComputeObjf(self):

    def Dim(self):
        return self.stats_.Dim()
    # E-step
    # def EstimateOneIter(self):

    # def InitParameters(self):

    # def ResetPerIterStats(self):

    # gets stats from intra-class variation (stats_.offset_scatter_).
    # def GetStatsFromIntraClass(self):

    # gets part of stats relating to class means.
    def GetStatsFromClassMeans(self):
        # SpMatrix<double> between_var_inv(between_var_);
        # between_var_inv.Invert();
        between_var_inv = np.linalg.inv(self.between_var_)

        #   SpMatrix<double> within_var_inv(within_var_);
        #   within_var_inv.Invert();
        within_var_inv = np.linalg.inv(self.within_var_)

        # mixed_var will equal (between_var^{-1} + n within_var^{-1})^{-1}.
        mixed_var = np.array(self.Dim())
        n = -1 # the current number of examples for the class.

        # for (size_t i = 0; i < stats_.class_info_.size(); i++) {
        # todo
        # for i in range(self.stats_.class_info_.size()):

        #     const ClassInfo &info = stats_.class_info_[i];
        #     double weight = info.weight;
        #     if (info.num_examples != n) {
        #       n = info.num_examples;
        #       mixed_var.CopyFromSp(between_var_inv);
        #       mixed_var.AddSp(n, within_var_inv);
        #       mixed_var.Invert();
        #     }
        #     Vector<double> m = *(info.mean); // the mean for this class.
        #     m.AddVec(-1.0 / stats_.class_weight_, stats_.sum_); // remove global mean
        #     Vector<double> temp(Dim()); // n within_var^{-1} m
        #     temp.AddSpVec(n, within_var_inv, m, 0.0);
        #     Vector<double> w(Dim()); // w, as defined in the comment.
        #     w.AddSpVec(1.0, mixed_var, temp, 0.0);
        #     Vector<double> m_w(m); // m - w
        #     m_w.AddVec(-1.0, w);
        #     between_var_stats_.AddSp(weight, mixed_var);
        #     between_var_stats_.AddVec2(weight, w);
        #     between_var_count_ += weight;
        #     within_var_stats_.AddSp(weight * n, mixed_var);
        #     within_var_stats_.AddVec2(weight * n, m_w);
        #     within_var_count_ += weight;
        #   }


    # M-step
    # def EstimateFromStats(self):
    #
    # # Copy to output.
    # def GetOutput(Plda *plda);
    #


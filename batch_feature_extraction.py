# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameter

process_str = 'dev, eval'   # 'dev' or 'eval' will extract features for the respective set accordingly
                            #  'dev, eval' will extract features of both sets together

params = parameter.get_params()


if 'dev' in process_str:
    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

    # Extract features and normalize them
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # # Extract labels in regression mode
    dev_feat_cls.extract_all_labels()


if 'eval' in process_str:
    # -----------------------------Extract ONLY features for evaluation set-----------------------------
    eval_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)

    # Extract features and normalize them
    eval_feat_cls.extract_all_feature()
    eval_feat_cls.preprocess_features()


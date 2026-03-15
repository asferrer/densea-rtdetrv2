#!/usr/bin/env python3
"""
Collect results from all experiments and generate comparison visualizations.

This script:
1. Extracts mAP and per-class AP from evaluation outputs
2. Creates a comparison table (CSV)
3. Generates visualizations:
   - mAP vs real data percentage
   - Per-class AP comparison
   - Minority class analysis
4. Generates a markdown report with conclusions

Usage:
    python experiments/collect_results.py
"""

import json
import re
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from config import (
    EXPERIMENTS,
    REAL_ONLY_EXPERIMENTS,
    ALL_EXPERIMENTS,
    COMPARISON_PAIRS,
    EVALUATION_OUTPUT_DIR,
    RESULTS_OUTPUT_DIR,
    TRAINING_OUTPUT_DIR,
    CLASSES,
    CLASS_NAMES,
    MINORITY_CLASSES,
    REAL_CLASS_COUNTS,
    # Versioned dataset support
    SYNTHETIC_DATASETS,
    DEFAULT_SYNTH_VERSION,
    get_synth_config,
    get_available_synth_versions,
    get_versioned_output_paths,
    experiment_uses_synthetic,
    get_experiment_type,
    get_experiment_output_paths,
    get_real_only_output_paths,
    # Batch mode support
    BATCH_MODES,
    DEFAULT_BATCH_MODE,
)

# =============================================================================
# INTERNATIONALIZATION (i18n)
# =============================================================================

TRANSLATIONS = {
    'en': {
        'title': 'Comprehensive Experiment Report',
        'subtitle': 'Synthetic vs Real Data Analysis for RT-DETRv2 Marine Debris Detection',
        'generated': 'Generated',
        'toc': 'Table of Contents',
        'executive_summary': 'Executive Summary',
        'methodology': 'Methodology',
        'training_overview': 'Training Overview',
        'key_visualizations': 'Key Visualizations',
        'complete_results': 'Complete Results Table',
        'detailed_analysis': 'Detailed Experiment Analysis',
        'per_class_analysis': 'Per-Class Performance Analysis',
        'error_analysis': 'Error Analysis',
        'size_analysis': 'Object Size Analysis',
        'conclusions': 'Conclusions',
        'recommendations': 'Recommendations',
        'appendix_cm': 'Appendix: All Confusion Matrices',
        'appendix_scores': 'Appendix: Score Distributions',
        'files_reference': 'Generated Files Reference',
        'key_results': 'Key Results at a Glance',
        'critical_findings': 'Critical Findings',
        'best_model': 'Best Performing Model',
        'real_only_baseline': 'Real-Only Baseline',
        'synth_only_baseline': 'Synthetic-Only Baseline',
        'improvement_real': 'Improvement over Real-Only',
        'improvement_synth': 'Improvement over Synth-Only',
        'total_training_time': 'Total Training Time',
        'num_experiments': 'Number of Experiments',
        'object_classes': 'Object Classes',
        'domain_gap': 'Domain Gap Impact',
        'real_data_value': 'Real Data Value',
        'minority_strategy': 'Minority Class Strategy',
        'model_architecture': 'Model Architecture',
        'experiment_design': 'Experiment Design',
        'dataset_info': 'Dataset Information',
        'real_dataset_dist': 'Real Dataset Distribution (Training)',
        'training_config': 'Training Configuration',
        'max_epochs': 'Max Epochs',
        'early_stopping': 'Early Stopping',
        'batch_size': 'Batch Size',
        'eval_metric': 'Evaluation Metric',
        'training_duration': 'Training Duration by Experiment',
        'experiment': 'Experiment',
        'duration': 'Duration',
        'status': 'Status',
        'epochs': 'Epochs',
        'hours': 'hours',
        'days': 'days',
        'map_vs_real': 'Model Performance vs Real Data Percentage',
        'map_vs_real_desc': 'This chart shows how mAP improves as the proportion of real data increases (with 100% synthetic as base).',
        'key_observations': 'Key Observations',
        'minority_comparison': 'Minority Class Performance Comparison',
        'minority_desc': 'Performance of underrepresented classes (Can, Metal_Debris) across all experiments.',
        'per_class_comparison': 'Per-Class AP Comparison',
        'per_class_desc': 'Comparison of AP values for all 8 classes across key experiments.',
        'primary_metrics': 'Primary Metrics',
        'per_class_ap': 'Per-Class AP (mAP@0.5:0.95)',
        'type': 'Type',
        'description': 'Description',
        'training_info': 'Training Information',
        'final_loss': 'Final Loss',
        'eval_metrics': 'Evaluation Metrics',
        'detection_stats': 'Detection Statistics',
        'total_gt': 'Total GT Objects',
        'total_pred': 'Total Predictions',
        'true_positives': 'True Positives',
        'false_negatives': 'False Negatives',
        'false_positives': 'False Positives',
        'per_class_detection': 'Per-Class Detection Performance',
        'click_expand_cm': 'Click to expand Confusion Matrix',
        'click_expand_ap': 'Click to expand Per-Class AP Chart',
        'click_expand_scores': 'Click to expand Score Distributions',
        'class_comparison': 'Class-wise mAP Comparison Across All Experiments',
        'minority_class': 'Minority Class',
        'fp_fn_analysis': 'False Positive and False Negative Analysis',
        'fp_fn_desc': 'This section analyzes the types of errors made by each model.',
        'size_variation': 'Performance variation by object size',
        'size_obs_small': 'Small objects are consistently the hardest to detect across all experiments',
        'size_obs_large': 'Large objects benefit most from synthetic data augmentation',
        'size_obs_medium': 'Real data is critical for medium-sized object detection',
        'hypothesis_validation': 'Hypothesis Validation',
        'synth_domain_gap': 'Synthetic Data Domain Gap',
        'real_data_impact': 'Real Data Impact',
        'minority_improvement': 'Minority Class Improvement',
        'performance_gap': 'Performance gap',
        'synth_achieves': 'Synthetic achieves',
        'of_real_performance': 'of real performance',
        'conclusion': 'Conclusion',
        'domain_gap_conclusion': 'The domain gap between synthetic and real images causes a',
        'performance_degradation': 'performance degradation. Synthetic data alone is',
        'insufficient': 'insufficient',
        'for_production': 'for production deployment.',
        'key_findings': 'Key Findings',
        'first_10_pct': 'First 10% of real data provides',
        'relative_improvement': 'relative improvement',
        'diminishing_returns': 'Diminishing returns observed after 50% real data',
        'combined_outperform': 'Combined datasets (synth+real) outperform real-only training',
        'targeted_synth': 'Targeted synthetic augmentation for minority classes improves their detection',
        'without_degrading': 'without degrading performance on majority classes. The',
        'best_balance': 'achieves the best overall balance.',
        'production_deployment': 'Production Deployment',
        'recommended_model': 'Recommended Model',
        'use_for_production': 'for production deployment',
        'best_balance_minority': 'Best balance between minority class detection and overall performance',
        'data_collection': 'Data Collection Priority',
        'focus_minority': 'Focus on collecting real samples for minority classes (Can, Metal_Debris)',
        'even_10_pct': 'Even 10% real data provides significant improvement over synthetic-only',
        'synth_strategy': 'Synthetic Data Strategy',
        'use_inverse_prop': 'Use inverse-proportional sampling for synthetic generation',
        'generate_more_synth': 'Generate more synthetic samples for underrepresented classes',
        'avoid_synth_only': 'Avoid synthetic-only training for production systems',
        'training_config_rec': 'Training Configuration',
        'use_combined': 'Use combined real + targeted synthetic data',
        'apply_early_stop': 'Apply early stopping with 20 epochs patience',
        'monitor_minority': 'Monitor minority class performance during training',
        'future_work': 'Future Work',
        'domain_adaptation': 'Domain Adaptation: Explore techniques to reduce synthetic-to-real domain gap',
        'active_learning': 'Active Learning: Prioritize labeling real samples from minority classes',
        'data_augmentation': 'Data Augmentation: Test additional augmentation strategies for minority classes',
        'model_ensemble': 'Model Ensemble: Consider ensembling models from different experiments',
        'main_results': 'Main Results',
        'aggregated_viz': 'Aggregated Visualizations',
        'per_exp_images': 'Per-Experiment Images (in `images/` subfolder)',
        'each_exp_contains': 'Each experiment folder contains',
        'auto_generated': 'Report generated automatically by collect_results.py',
        'synthetic_only': 'Synthetic Only',
        'real_only': 'Real Only',
        'minority_focused': 'Minority-Focused',
        'proportional': 'Proportional',
        'mixed': 'Mixed',
        'minority_indicator': 'Indicates minority classes (<2% of annotations)',
        'sharp_improvement': 'Sharp improvement from 0% to 10% real data',
        'diminishing_after_50': 'Diminishing returns after 50% real data',
        'real_only_underperform': 'Real-only (100%) slightly underperforms combined datasets',
        'both_benefit': 'Both classes benefit significantly from targeted synthetic augmentation',
        'exp08_best_can': 'exp08 (minority-only strategy) best for Can class',
        'exp09_best_metal': 'exp09 (proportional strategy) best for Metal_Debris',
        'executive_summary_intro': 'This report presents a comprehensive analysis of **9 experiments** evaluating the impact of synthetic-to-real data ratios on RT-DETRv2 object detection performance for marine debris.',
        'metric': 'Metric',
        'value': 'Value',
        'class': 'Class',
        'samples': 'Samples',
        'percentage': 'Percentage',
        'total': 'Total',
        'ap_improvement': 'AP Improvement',
        'real_data_pct': 'Real Data %',
        'delta_map': 'Δ mAP',
        'file': 'File',
        'this_report': 'This comprehensive report',
        'complete_metrics_table': 'Complete metrics table for all experiments',
        'performance_progression': 'Performance progression chart',
        'minority_class_comparison': 'Minority class (Can, Metal_Debris) comparison',
        'per_class_key_exp': 'Per-class AP across key experiments',
        'norm_conf_matrix': 'Normalized confusion matrix',
        'raw_conf_matrix': 'Raw confusion matrix',
        'per_class_ap_img': 'Per-class Average Precision',
        'conf_score_dist': 'Confidence score distributions',
        'fp_fn_img': 'False positive/negative analysis',
        'ap_ar_size_img': 'AP/AR by object size',
        'train_dist_img': 'Training set class distribution',
        'val_dist_img': 'Validation set class distribution',
        'report_auto_gen': 'Report generated automatically by collect_results.py',
        'input_resolution': 'Input Resolution',
        'loss_functions': 'Loss Functions',
        'optimizer': 'Optimizer',
        'pretrained': 'pretrained on COCO',
        'patience_epochs': 'Patience of {n} epochs',
        'monitoring': 'monitoring',
        # Paired comparison translations
        'paired_comparison': 'Paired Comparison: Real-Only vs Synthetic+Real',
        'paired_comparison_desc': 'This chart directly compares training with X% real data only versus training with 100% synthetic + X% real data.',
        'synth_benefit_analysis': 'Synthetic Data Benefit Analysis',
        'synth_benefit_desc': 'Shows the mAP improvement from adding synthetic data at each real data level.',
        'paired_summary_table': 'Paired Comparison Summary',
        'real_only_map': 'Real-Only mAP',
        'synth_real_map': 'Synth+Real mAP',
        'improvement': 'Improvement',
        'synth_most_benefit_low': 'Synthetic data provides most benefit at lower real data percentages',
        'gap_narrows': 'The gap narrows as real data percentage increases',
        'diminishing_100': 'At 100% real data, synthetic may provide diminishing or negative returns',
        # Cross-version comparison translations
        'cv_title': 'Cross-Version Synthetic Dataset Comparison Report',
        'cv_subtitle': 'Comparative Analysis of Synthetic Dataset Versions for RT-DETRv2 Marine Debris Detection',
        'cv_toc': 'Table of Contents',
        'cv_executive_summary': 'Executive Summary',
        'cv_versions_compared': 'Synthetic Dataset Versions Compared',
        'cv_methodology': 'Methodology',
        'cv_map_comparison': 'Overall mAP Comparison',
        'cv_map_curves': 'Performance Curves: mAP vs Real Data Ratio',
        'cv_per_class_analysis': 'Per-Class Performance Analysis',
        'cv_minority_analysis': 'Minority Class Analysis',
        'cv_domain_gap_analysis': 'Domain Gap Analysis',
        'cv_size_analysis': 'Object Size Analysis',
        'cv_training_efficiency': 'Training Efficiency',
        'cv_synth_benefit': 'Synthetic Data Benefit by Version',
        'cv_radar_chart': 'Multi-Metric Radar Comparison',
        'cv_ranking': 'Version Ranking',
        'cv_conclusions': 'Conclusions',
        'cv_recommendations': 'Recommendations',
        'cv_files_reference': 'Generated Files Reference',
        'cv_best_version_overall': 'Best Overall Version',
        'cv_wins_by_version': 'Wins by Version',
        'cv_avg_map': 'Average mAP',
        'cv_completed_exps': 'Completed Experiments',
        'cv_per_class_champion': 'Per-Class Champion',
        'cv_domain_gap_desc': 'The domain gap measures the performance drop when training with synthetic-only data vs real-only data. A smaller gap indicates better synthetic data quality.',
        'cv_synth_only_map': 'Synth-Only mAP (exp01)',
        'cv_real_only_map': 'Real-Only mAP (exp07)',
        'cv_gap_absolute': 'Absolute Gap',
        'cv_gap_relative': 'Relative Gap (%)',
        'cv_recovery_ratio': 'Recovery Ratio (exp06/exp07)',
        'cv_real_ratio_progression': 'mAP progression as real data percentage increases, compared across versions',
        'cv_version_strengths': 'Strengths',
        'cv_version_weaknesses': 'Weaknesses',
        'cv_num_classes': 'Num. Classes',
        'cv_missing_classes': 'Missing Classes',
        'cv_key_insight': 'Key Insight',
        'cv_surprising_finding': 'Notable Finding',
        'cv_trade_off': 'Trade-off',
        'cv_no_data': 'No data available',
        'cv_improvement_margin': 'Average Margin of Victory',
        'cv_dominant_version': 'Dominant version across most experiments',
        'cv_competitive_versions': 'Competitive versions with trade-offs',
        'cv_recommendation_production': 'For production deployment, use',
        'cv_recommendation_minority': 'For minority class improvement, use',
        'cv_recommendation_efficiency': 'For training efficiency, use',
        'cv_future_more_versions': 'Evaluate additional synthetic dataset versions to confirm findings',
        'cv_future_class_alignment': 'Investigate impact of class alignment on domain gap',
        'cv_future_ensemble': 'Consider ensemble models trained on different synthetic versions',
        'cv_future_domain_adapt': 'Apply domain adaptation techniques tailored to each synthetic source',
        'cv_report_auto_gen': 'Report generated automatically by collect_results.py',
        'cv_best_for_class': 'Best version for',
        'cv_synth_benefit_desc': 'Improvement in mAP from adding synthetic data at each real data level, compared across versions',
        'cv_training_hours': 'Avg. Training Hours',
        'cv_performance_ratio': 'mAP per Training Hour',
        'cv_exp_design': 'Experiment Design',
        'cv_exp_design_desc': 'All versions are evaluated using the same 9 experiments varying synthetic/real data ratios, plus 4 real-only baselines.',
        'cv_hypothesis_classes': 'Hypothesis: More synthetic classes lead to better performance',
        'cv_hypothesis_alignment': 'Hypothesis: Full class alignment reduces domain gap',
        'cv_conclusion_dominant': 'dominates across the majority of experiments',
        'cv_conclusion_competitive': 'shows competitive results in specific scenarios',
        'cv_conclusion_domain_gap': 'The version with more classes and better alignment achieves a smaller domain gap',
        'cv_conclusion_minority': 'For minority class detection',
        'cv_overall_winner': 'Overall Winner',
        'cv_margin': 'Margin',
        'cv_ap_small': 'AP Small',
        'cv_ap_medium': 'AP Medium',
        'cv_ap_large': 'AP Large',
        'cv_version': 'Version',
        'cv_name': 'Name',
        'cv_classes': 'Classes',
        'cv_none': 'None',
        'cv_best': 'Best',
        'cv_generated': 'Generated',
        'cv_paired_comparison': 'Paired Comparison: Real-Only vs Synthetic+Real',
        'cv_paired_desc': 'Direct comparison of training with only real data versus adding synthetic data from each version, at matched real data levels.',
        'cv_paired_obs_positive': 'Adding synthetic data improves performance at this level',
        'cv_paired_obs_negative': 'Synthetic data hurts performance at this level (domain gap dominates)',
        'cv_real_only_baseline': 'Real-Only Baseline',
        'cv_delta': 'Δ mAP',
        'cv_delta_vs_real': 'Δ vs Real',
        'cv_synth_real_label': 'Synth+Real',
        'cv_real_only_comparison_note': 'Experiments with a real-only counterpart show the direct delta (Δ) of adding synthetic data.',
        'cv_real_only_curve_note': 'The Real-Only curve shows training with only real data at each percentage level, serving as baseline to evaluate the impact of synthetic data.',
        'cv_impact_of_synthetic': 'Impact of Adding Synthetic Data',
        'cv_impact_positive': 'Positive impact: synthetic data helps',
        'cv_impact_negative': 'Negative impact: domain gap outweighs benefit',
        'cv_paired_conclusion': 'Paired Comparison Conclusion',
    },
    'es': {
        'title': 'Informe Completo de Experimentos',
        'subtitle': 'Análisis de Datos Sintéticos vs Reales para RT-DETRv2 en Detección de Residuos Marinos',
        'generated': 'Generado',
        'toc': 'Tabla de Contenidos',
        'executive_summary': 'Resumen Ejecutivo',
        'methodology': 'Metodología',
        'training_overview': 'Resumen del Entrenamiento',
        'key_visualizations': 'Visualizaciones Clave',
        'complete_results': 'Tabla Completa de Resultados',
        'detailed_analysis': 'Análisis Detallado de Experimentos',
        'per_class_analysis': 'Análisis de Rendimiento por Clase',
        'error_analysis': 'Análisis de Errores',
        'size_analysis': 'Análisis por Tamaño de Objeto',
        'conclusions': 'Conclusiones',
        'recommendations': 'Recomendaciones',
        'appendix_cm': 'Apéndice: Todas las Matrices de Confusión',
        'appendix_scores': 'Apéndice: Distribuciones de Puntuación',
        'files_reference': 'Referencia de Archivos Generados',
        'key_results': 'Resultados Clave',
        'critical_findings': 'Hallazgos Críticos',
        'best_model': 'Mejor Modelo',
        'real_only_baseline': 'Línea Base Solo Real',
        'synth_only_baseline': 'Línea Base Solo Sintético',
        'improvement_real': 'Mejora vs Solo Real',
        'improvement_synth': 'Mejora vs Solo Sintético',
        'total_training_time': 'Tiempo Total de Entrenamiento',
        'num_experiments': 'Número de Experimentos',
        'object_classes': 'Clases de Objetos',
        'domain_gap': 'Impacto del Domain Gap',
        'real_data_value': 'Valor de los Datos Reales',
        'minority_strategy': 'Estrategia para Clases Minoritarias',
        'model_architecture': 'Arquitectura del Modelo',
        'experiment_design': 'Diseño Experimental',
        'dataset_info': 'Información del Dataset',
        'real_dataset_dist': 'Distribución del Dataset Real (Entrenamiento)',
        'training_config': 'Configuración del Entrenamiento',
        'max_epochs': 'Épocas Máximas',
        'early_stopping': 'Parada Temprana',
        'batch_size': 'Tamaño de Batch',
        'eval_metric': 'Métrica de Evaluación',
        'training_duration': 'Duración del Entrenamiento por Experimento',
        'experiment': 'Experimento',
        'duration': 'Duración',
        'status': 'Estado',
        'epochs': 'Épocas',
        'hours': 'horas',
        'days': 'días',
        'map_vs_real': 'Rendimiento del Modelo vs Porcentaje de Datos Reales',
        'map_vs_real_desc': 'Este gráfico muestra cómo mejora el mAP a medida que aumenta la proporción de datos reales (con 100% sintético como base).',
        'key_observations': 'Observaciones Clave',
        'minority_comparison': 'Comparación de Rendimiento en Clases Minoritarias',
        'minority_desc': 'Rendimiento de las clases subrepresentadas (Can, Metal_Debris) en todos los experimentos.',
        'per_class_comparison': 'Comparación de AP por Clase',
        'per_class_desc': 'Comparación de valores AP para las 8 clases en experimentos clave.',
        'primary_metrics': 'Métricas Principales',
        'per_class_ap': 'AP por Clase (mAP@0.5:0.95)',
        'type': 'Tipo',
        'description': 'Descripción',
        'training_info': 'Información del Entrenamiento',
        'final_loss': 'Loss Final',
        'eval_metrics': 'Métricas de Evaluación',
        'detection_stats': 'Estadísticas de Detección',
        'total_gt': 'Total Objetos GT',
        'total_pred': 'Total Predicciones',
        'true_positives': 'Verdaderos Positivos',
        'false_negatives': 'Falsos Negativos',
        'false_positives': 'Falsos Positivos',
        'per_class_detection': 'Rendimiento de Detección por Clase',
        'click_expand_cm': 'Click para expandir Matriz de Confusión',
        'click_expand_ap': 'Click para expandir Gráfico AP por Clase',
        'click_expand_scores': 'Click para expandir Distribuciones de Puntuación',
        'class_comparison': 'Comparación de mAP por Clase en Todos los Experimentos',
        'minority_class': 'Clase Minoritaria',
        'fp_fn_analysis': 'Análisis de Falsos Positivos y Falsos Negativos',
        'fp_fn_desc': 'Esta sección analiza los tipos de errores cometidos por cada modelo.',
        'size_variation': 'Variación del rendimiento por tamaño de objeto',
        'size_obs_small': 'Los objetos pequeños son consistentemente los más difíciles de detectar en todos los experimentos',
        'size_obs_large': 'Los objetos grandes se benefician más del aumento de datos sintéticos',
        'size_obs_medium': 'Los datos reales son críticos para la detección de objetos medianos',
        'hypothesis_validation': 'Validación de Hipótesis',
        'synth_domain_gap': 'Domain Gap de Datos Sintéticos',
        'real_data_impact': 'Impacto de los Datos Reales',
        'minority_improvement': 'Mejora en Clases Minoritarias',
        'performance_gap': 'Brecha de rendimiento',
        'synth_achieves': 'Sintético alcanza',
        'of_real_performance': 'del rendimiento real',
        'conclusion': 'Conclusión',
        'domain_gap_conclusion': 'El domain gap entre imágenes sintéticas y reales causa una',
        'performance_degradation': 'degradación del rendimiento. Los datos sintéticos solos son',
        'insufficient': 'insuficientes',
        'for_production': 'para despliegue en producción.',
        'key_findings': 'Hallazgos Clave',
        'first_10_pct': 'El primer 10% de datos reales proporciona',
        'relative_improvement': 'mejora relativa',
        'diminishing_returns': 'Rendimientos decrecientes observados después del 50% de datos reales',
        'combined_outperform': 'Los datasets combinados (sint+real) superan al entrenamiento solo con datos reales',
        'targeted_synth': 'El aumento sintético dirigido para clases minoritarias mejora su detección',
        'without_degrading': 'sin degradar el rendimiento en las clases mayoritarias. La estrategia',
        'best_balance': 'logra el mejor equilibrio general.',
        'production_deployment': 'Despliegue en Producción',
        'recommended_model': 'Modelo Recomendado',
        'use_for_production': 'para despliegue en producción',
        'best_balance_minority': 'Mejor equilibrio entre detección de clases minoritarias y rendimiento general',
        'data_collection': 'Prioridad de Recolección de Datos',
        'focus_minority': 'Enfocarse en recolectar muestras reales para clases minoritarias (Can, Metal_Debris)',
        'even_10_pct': 'Incluso 10% de datos reales proporciona mejora significativa vs solo sintético',
        'synth_strategy': 'Estrategia de Datos Sintéticos',
        'use_inverse_prop': 'Usar muestreo inversamente proporcional para generación sintética',
        'generate_more_synth': 'Generar más muestras sintéticas para clases subrepresentadas',
        'avoid_synth_only': 'Evitar entrenamiento solo con sintéticos para sistemas de producción',
        'training_config_rec': 'Configuración del Entrenamiento',
        'use_combined': 'Usar datos combinados reales + sintéticos dirigidos',
        'apply_early_stop': 'Aplicar parada temprana con paciencia de 40 épocas',
        'monitor_minority': 'Monitorear rendimiento de clases minoritarias durante entrenamiento',
        'future_work': 'Trabajo Futuro',
        'domain_adaptation': 'Adaptación de Dominio: Explorar técnicas para reducir el domain gap sintético-real',
        'active_learning': 'Aprendizaje Activo: Priorizar etiquetado de muestras reales de clases minoritarias',
        'data_augmentation': 'Aumento de Datos: Probar estrategias adicionales de aumento para clases minoritarias',
        'model_ensemble': 'Ensemble de Modelos: Considerar ensemble de modelos de diferentes experimentos',
        'main_results': 'Resultados Principales',
        'aggregated_viz': 'Visualizaciones Agregadas',
        'per_exp_images': 'Imágenes por Experimento (en subcarpeta `images/`)',
        'each_exp_contains': 'Cada carpeta de experimento contiene',
        'auto_generated': 'Informe generado automáticamente por collect_results.py',
        'synthetic_only': 'Solo Sintético',
        'real_only': 'Solo Real',
        'minority_focused': 'Enfocado en Minoritarias',
        'proportional': 'Proporcional',
        'mixed': 'Mixto',
        'minority_indicator': 'Indica clases minoritarias (<2% de anotaciones)',
        'sharp_improvement': 'Mejora pronunciada de 0% a 10% de datos reales',
        'diminishing_after_50': 'Rendimientos decrecientes después del 50% de datos reales',
        'real_only_underperform': 'Solo real (100%) rinde ligeramente menos que datasets combinados',
        'both_benefit': 'Ambas clases se benefician significativamente del aumento sintético dirigido',
        'exp08_best_can': 'exp08 (estrategia solo minoritarias) mejor para clase Can',
        'exp09_best_metal': 'exp09 (estrategia proporcional) mejor para Metal_Debris',
        'executive_summary_intro': 'Este informe presenta un análisis completo de **9 experimentos** que evalúan el impacto de las proporciones de datos sintéticos vs reales en el rendimiento de detección de objetos RT-DETRv2 para residuos marinos.',
        'metric': 'Métrica',
        'value': 'Valor',
        'class': 'Clase',
        'samples': 'Muestras',
        'percentage': 'Porcentaje',
        'total': 'Total',
        'ap_improvement': 'Mejora de AP',
        'real_data_pct': '% Datos Reales',
        'delta_map': 'Δ mAP',
        'file': 'Archivo',
        'this_report': 'Este informe completo',
        'complete_metrics_table': 'Tabla completa de métricas para todos los experimentos',
        'performance_progression': 'Gráfico de progresión de rendimiento',
        'minority_class_comparison': 'Comparación de clases minoritarias (Can, Metal_Debris)',
        'per_class_key_exp': 'AP por clase en experimentos clave',
        'norm_conf_matrix': 'Matriz de confusión normalizada',
        'raw_conf_matrix': 'Matriz de confusión sin normalizar',
        'per_class_ap_img': 'Precisión Promedio por clase',
        'conf_score_dist': 'Distribuciones de puntuación de confianza',
        'fp_fn_img': 'Análisis de falsos positivos/negativos',
        'ap_ar_size_img': 'AP/AR por tamaño de objeto',
        'train_dist_img': 'Distribución de clases en conjunto de entrenamiento',
        'val_dist_img': 'Distribución de clases en conjunto de validación',
        'report_auto_gen': 'Informe generado automáticamente por collect_results.py',
        'input_resolution': 'Resolución de Entrada',
        'loss_functions': 'Funciones de Pérdida',
        'optimizer': 'Optimizador',
        'pretrained': 'preentrenado en COCO',
        'patience_epochs': 'Paciencia de {n} épocas',
        'monitoring': 'monitoreando',
        # Paired comparison translations
        'paired_comparison': 'Comparación Pareada: Solo Real vs Sintético+Real',
        'paired_comparison_desc': 'Este gráfico compara directamente el entrenamiento con X% de datos reales vs 100% sintético + X% real.',
        'synth_benefit_analysis': 'Análisis del Beneficio de Datos Sintéticos',
        'synth_benefit_desc': 'Muestra la mejora de mAP al añadir datos sintéticos en cada nivel de datos reales.',
        'paired_summary_table': 'Resumen de Comparación Pareada',
        'real_only_map': 'mAP Solo Real',
        'synth_real_map': 'mAP Sint+Real',
        'improvement': 'Mejora',
        'synth_most_benefit_low': 'Los datos sintéticos aportan más beneficio con porcentajes bajos de datos reales',
        'gap_narrows': 'La diferencia se reduce a medida que aumenta el porcentaje de datos reales',
        'diminishing_100': 'Con 100% de datos reales, el sintético puede aportar rendimientos decrecientes o negativos',
        # Cross-version comparison translations
        'cv_title': 'Informe Comparativo de Versiones de Datasets Sintéticos',
        'cv_subtitle': 'Análisis Comparativo de Versiones de Datasets Sintéticos para RT-DETRv2 en Detección de Residuos Marinos',
        'cv_toc': 'Tabla de Contenidos',
        'cv_executive_summary': 'Resumen Ejecutivo',
        'cv_versions_compared': 'Versiones de Datasets Sintéticos Comparadas',
        'cv_methodology': 'Metodología',
        'cv_map_comparison': 'Comparación General de mAP',
        'cv_map_curves': 'Curvas de Rendimiento: mAP vs Ratio de Datos Reales',
        'cv_per_class_analysis': 'Análisis de Rendimiento por Clase',
        'cv_minority_analysis': 'Análisis de Clases Minoritarias',
        'cv_domain_gap_analysis': 'Análisis de Domain Gap',
        'cv_size_analysis': 'Análisis por Tamaño de Objeto',
        'cv_training_efficiency': 'Eficiencia de Entrenamiento',
        'cv_synth_benefit': 'Beneficio de Datos Sintéticos por Versión',
        'cv_radar_chart': 'Comparación Radar Multi-Métrica',
        'cv_ranking': 'Ranking de Versiones',
        'cv_conclusions': 'Conclusiones',
        'cv_recommendations': 'Recomendaciones',
        'cv_files_reference': 'Referencia de Archivos Generados',
        'cv_best_version_overall': 'Mejor Versión General',
        'cv_wins_by_version': 'Victorias por Versión',
        'cv_avg_map': 'mAP Promedio',
        'cv_completed_exps': 'Experimentos Completados',
        'cv_per_class_champion': 'Campeón por Clase',
        'cv_domain_gap_desc': 'El domain gap mide la caída de rendimiento al entrenar con datos solo sintéticos vs solo reales. Un gap menor indica mejor calidad de los datos sintéticos.',
        'cv_synth_only_map': 'mAP Solo Sintético (exp01)',
        'cv_real_only_map': 'mAP Solo Real (exp07)',
        'cv_gap_absolute': 'Gap Absoluto',
        'cv_gap_relative': 'Gap Relativo (%)',
        'cv_recovery_ratio': 'Ratio de Recuperación (exp06/exp07)',
        'cv_real_ratio_progression': 'Progresión de mAP a medida que aumenta el porcentaje de datos reales, comparado entre versiones',
        'cv_version_strengths': 'Fortalezas',
        'cv_version_weaknesses': 'Debilidades',
        'cv_num_classes': 'Núm. Clases',
        'cv_missing_classes': 'Clases Faltantes',
        'cv_key_insight': 'Hallazgo Clave',
        'cv_surprising_finding': 'Hallazgo Notable',
        'cv_trade_off': 'Compensación',
        'cv_no_data': 'Sin datos disponibles',
        'cv_improvement_margin': 'Margen Promedio de Victoria',
        'cv_dominant_version': 'Versión dominante en la mayoría de experimentos',
        'cv_competitive_versions': 'Versiones competitivas con compensaciones',
        'cv_recommendation_production': 'Para despliegue en producción, usar',
        'cv_recommendation_minority': 'Para mejora de clases minoritarias, usar',
        'cv_recommendation_efficiency': 'Para eficiencia de entrenamiento, usar',
        'cv_future_more_versions': 'Evaluar versiones adicionales de datasets sintéticos para confirmar hallazgos',
        'cv_future_class_alignment': 'Investigar el impacto de la alineación de clases en el domain gap',
        'cv_future_ensemble': 'Considerar ensemble de modelos entrenados con diferentes versiones sintéticas',
        'cv_future_domain_adapt': 'Aplicar técnicas de adaptación de dominio específicas para cada fuente sintética',
        'cv_report_auto_gen': 'Informe generado automáticamente por collect_results.py',
        'cv_best_for_class': 'Mejor versión para',
        'cv_synth_benefit_desc': 'Mejora en mAP al añadir datos sintéticos en cada nivel de datos reales, comparado entre versiones',
        'cv_training_hours': 'Horas Prom. Entrenamiento',
        'cv_performance_ratio': 'mAP por Hora de Entrenamiento',
        'cv_exp_design': 'Diseño Experimental',
        'cv_exp_design_desc': 'Todas las versiones se evalúan usando los mismos 9 experimentos variando ratios de datos sintéticos/reales, más 4 líneas base solo reales.',
        'cv_hypothesis_classes': 'Hipótesis: Más clases sintéticas llevan a mejor rendimiento',
        'cv_hypothesis_alignment': 'Hipótesis: Alineación completa de clases reduce el domain gap',
        'cv_conclusion_dominant': 'domina en la mayoría de los experimentos',
        'cv_conclusion_competitive': 'muestra resultados competitivos en escenarios específicos',
        'cv_conclusion_domain_gap': 'La versión con más clases y mejor alineación logra un menor domain gap',
        'cv_conclusion_minority': 'Para detección de clases minoritarias',
        'cv_overall_winner': 'Ganador General',
        'cv_margin': 'Margen',
        'cv_ap_small': 'AP Pequeño',
        'cv_ap_medium': 'AP Mediano',
        'cv_ap_large': 'AP Grande',
        'cv_version': 'Versión',
        'cv_name': 'Nombre',
        'cv_classes': 'Clases',
        'cv_none': 'Ninguna',
        'cv_best': 'Mejor',
        'cv_generated': 'Generado',
        'cv_paired_comparison': 'Comparación Pareada: Solo Real vs Sintético+Real',
        'cv_paired_desc': 'Comparación directa del entrenamiento con solo datos reales versus añadir datos sintéticos de cada versión, a niveles equivalentes de datos reales.',
        'cv_paired_obs_positive': 'Añadir datos sintéticos mejora el rendimiento a este nivel',
        'cv_paired_obs_negative': 'Los datos sintéticos perjudican el rendimiento a este nivel (el domain gap domina)',
        'cv_real_only_baseline': 'Línea Base Solo Real',
        'cv_delta': 'Δ mAP',
        'cv_delta_vs_real': 'Δ vs Real',
        'cv_synth_real_label': 'Sintético+Real',
        'cv_real_only_comparison_note': 'Los experimentos con contraparte solo-real muestran el delta (Δ) directo de añadir datos sintéticos.',
        'cv_real_only_curve_note': 'La curva Solo Real muestra el entrenamiento con solo datos reales a cada nivel porcentual, sirviendo como línea base para evaluar el impacto de los datos sintéticos.',
        'cv_impact_of_synthetic': 'Impacto de Añadir Datos Sintéticos',
        'cv_impact_positive': 'Impacto positivo: los datos sintéticos ayudan',
        'cv_impact_negative': 'Impacto negativo: el domain gap supera el beneficio',
        'cv_paired_conclusion': 'Conclusión de la Comparación Pareada',
    }
}


def t(key: str, lang: str = 'en') -> str:
    """Get translation for a key in the specified language."""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# Try to import visualization libraries
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not available. CSV output will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. Visualizations will be skipped.")


# =============================================================================
# METRICS EXTRACTION
# =============================================================================

def extract_coco_metrics(eval_dir: Path) -> Dict[str, float]:
    """
    Extract COCO evaluation metrics from evaluation output.

    Looks for:
    - coco_evaluation_report.txt
    - per_class_ap_report.txt
    - complete_detection_report.txt

    Args:
        eval_dir: Path to evaluation output directory

    Returns:
        Dict with extracted metrics
    """
    metrics = {}

    # Check if reports are in 'best' subdirectory (from evaluation script)
    best_dir = eval_dir / "best"
    search_dir = best_dir if best_dir.exists() else eval_dir

    # Try to find COCO evaluation summary
    coco_report = search_dir / "coco_evaluation_report.txt"
    if coco_report.exists():
        with open(coco_report, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract mAP values using regex
        # Format: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
        patterns = {
            'mAP': r'AP.*IoU=0\.50:0\.95.*all.*=\s*([\d.]+)',
            'mAP_50': r'AP.*IoU=0\.50\s.*all.*=\s*([\d.]+)',
            'mAP_75': r'AP.*IoU=0\.75\s.*all.*=\s*([\d.]+)',
            'AP_small': r'AP.*small.*=\s*([\d.]+)',
            'AP_medium': r'AP.*medium.*=\s*([\d.]+)',
            'AP_large': r'AP.*large.*=\s*([\d.]+)',
            'AR_1': r'AR.*maxDets=  1.*=\s*([\d.]+)',
            'AR_10': r'AR.*maxDets= 10.*=\s*([\d.]+)',
            'AR_100': r'AR.*maxDets=100.*=\s*([\d.]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1))

    # Try to find per-class AP
    per_class_report = search_dir / "per_class_ap_report.txt"
    if per_class_report.exists():
        with open(per_class_report, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract per-class AP values
        # Format: Clase                 AP@0.5:0.95      AP@0.50      AP@0.75
        #         Bottle                     0.0246       0.1211       0.0012
        for class_name in CLASS_NAMES:
            # Match class name followed by whitespace and three numeric values
            pattern = rf'{class_name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metrics[f'AP_{class_name}'] = float(match.group(1))  # AP@0.5:0.95
                metrics[f'AP50_{class_name}'] = float(match.group(2))  # AP@0.50
                metrics[f'AP75_{class_name}'] = float(match.group(3))  # AP@0.75

    # Try complete detection report as fallback
    complete_report = search_dir / "complete_detection_report.txt"
    if complete_report.exists() and 'mAP' not in metrics:
        with open(complete_report, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract mAP
        match = re.search(r'mAP[@\s]*0\.5:0\.95[:\s]*([\d.]+)', content)
        if match:
            metrics['mAP'] = float(match.group(1))

    # Try to load JSON results if available
    json_files = list(search_dir.glob("*results*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if 'mAP' in data:
                        metrics['mAP'] = data['mAP']
                    if 'AP' in data:
                        metrics['mAP'] = data['AP']
        except:
            pass

    return metrics


def extract_training_info(
    exp_name: str,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Optional[Dict[str, Any]]:
    """Extract training information from experiment_state.json and training logs.

    Args:
        exp_name: Experiment name
        synth_version: Synthetic dataset version for versioned paths
        batch_mode: Batch sampling mode
    """
    # Get versioned paths based on experiment type
    exp = next((e for e in ALL_EXPERIMENTS if e['name'] == exp_name), None)
    if exp and experiment_uses_synthetic(exp):
        versioned_paths = get_versioned_output_paths(synth_version, batch_mode)
        state_file = Path(versioned_paths['results']) / f"experiment_state_{synth_version}.json"
        training_dir = Path(versioned_paths['training'])
    else:
        # Real-only experiment - use shared paths
        real_only_paths = get_real_only_output_paths(exp_name, batch_mode)
        state_file = Path(RESULTS_OUTPUT_DIR) / "experiment_state.json"
        training_dir = real_only_paths['training'].parent

    info = {}

    # Try to load from experiment state
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if exp_name in state.get('results', {}):
                exp_state = state['results'][exp_name]
                info['duration_formatted'] = exp_state.get('duration_formatted', 'N/A')
                info['duration_seconds'] = exp_state.get('duration_seconds', 0)
                info['completed_at'] = exp_state.get('completed_at', 'N/A')
                info['status'] = exp_state.get('status', 'unknown')
        except Exception as e:
            print(f"  Warning: Could not load training info for {exp_name}: {e}")

    # Try to extract from train.log
    train_log = training_dir / exp_name / "train.log"
    if train_log.exists():
        try:
            with open(train_log, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract final epoch
            epochs = re.findall(r'Epoch:\s*\[(\d+)\]', content)
            if epochs:
                info['total_epochs'] = max(int(e) for e in epochs) + 1

            # Extract best mAP from validation
            maps = re.findall(r'coco_eval_bbox.*?(\d+\.\d+)', content)
            if maps:
                info['best_val_map'] = max(float(m) for m in maps)

            # Extract final loss values
            losses = re.findall(r'loss:\s*([\d.]+)', content[-50000:])  # Last 50k chars
            if losses:
                info['final_loss'] = float(losses[-1])

        except Exception as e:
            print(f"  Warning: Could not parse train.log for {exp_name}: {e}")

    return info if info else None


def extract_score_distribution(
    exp_name: str,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Optional[Dict[str, Any]]:
    """Extract score distribution data from evaluation output."""
    # Get versioned paths based on experiment type
    exp = next((e for e in ALL_EXPERIMENTS if e['name'] == exp_name), None)
    if exp and experiment_uses_synthetic(exp):
        paths = get_experiment_output_paths(exp_name, synth_version, batch_mode)
        eval_dir = paths['evaluation'] / "best"
    else:
        paths = get_real_only_output_paths(exp_name, batch_mode)
        eval_dir = paths['evaluation'] / "best"
    score_file = eval_dir / "score_distribution_report.txt"

    if not score_file.exists():
        return None

    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            content = f.read()

        scores = {}
        pattern = r'(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)'
        for match in re.finditer(pattern, content):
            class_name = match.group(1)
            if class_name in CLASS_NAMES:
                scores[class_name] = {
                    'total': int(match.group(2)),
                    'above_0.5': int(match.group(3)),
                    'above_0.25': int(match.group(4)),
                    'above_0.1': int(match.group(5)),
                    'mean_score': float(match.group(6)),
                }
        return scores
    except Exception:
        return None


def extract_ap_by_size(
    exp_name: str,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Optional[Dict[str, float]]:
    """Extract AP/AR by object size from evaluation output."""
    # Get versioned paths based on experiment type
    exp = next((e for e in ALL_EXPERIMENTS if e['name'] == exp_name), None)
    if exp and experiment_uses_synthetic(exp):
        paths = get_experiment_output_paths(exp_name, synth_version, batch_mode)
        eval_dir = paths['evaluation'] / "best"
    else:
        paths = get_real_only_output_paths(exp_name, batch_mode)
        eval_dir = paths['evaluation'] / "best"
    size_file = eval_dir / "ap_by_size_report.txt"

    if not size_file.exists():
        return None

    try:
        with open(size_file, 'r', encoding='utf-8') as f:
            content = f.read()

        size_metrics = {}
        patterns = {
            'AP_small': r'AP \(small.*?\)\s+([\d.]+)',
            'AP_medium': r'AP \(medium.*?\)\s+([\d.]+)',
            'AP_large': r'AP \(large.*?\)\s+([\d.]+)',
            'AR_small': r'AR \(small.*?\)\s+([\d.]+)',
            'AR_medium': r'AR \(medium.*?\)\s+([\d.]+)',
            'AR_large': r'AR \(large.*?\)\s+([\d.]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                size_metrics[key] = float(match.group(1))

        return size_metrics
    except Exception:
        return None


def extract_fp_fn_analysis(
    exp_name: str,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Optional[Dict[str, Dict[str, int]]]:
    """Extract FP/FN analysis from evaluation output."""
    # Get versioned paths based on experiment type
    exp = next((e for e in ALL_EXPERIMENTS if e['name'] == exp_name), None)
    if exp and experiment_uses_synthetic(exp):
        paths = get_experiment_output_paths(exp_name, synth_version, batch_mode)
        eval_dir = paths['evaluation'] / "best"
    else:
        paths = get_real_only_output_paths(exp_name, batch_mode)
        eval_dir = paths['evaluation'] / "best"
    fp_fn_file = eval_dir / "fp_fn_report.txt"

    if not fp_fn_file.exists():
        return None

    try:
        with open(fp_fn_file, 'r', encoding='utf-8') as f:
            content = f.read()

        fp_fn = {}
        pattern = r'(\w+)\s+(\d+)\s+(\d+)'
        for match in re.finditer(pattern, content):
            class_name = match.group(1)
            if class_name in CLASS_NAMES:
                fp_fn[class_name] = {
                    'fp': int(match.group(2)),
                    'fn': int(match.group(3)),
                }
        return fp_fn
    except Exception:
        return None


def collect_all_metrics(
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, Dict]:
    """
    Collect metrics from all experiments including training info,
    score distributions, AP by size, and FP/FN analysis.

    Args:
        synth_version: Synthetic dataset version for versioned paths
        batch_mode: Batch sampling mode

    Returns:
        Dict mapping experiment name to comprehensive metrics dict
    """
    all_metrics = {}

    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']

        # Get versioned paths based on experiment type
        if experiment_uses_synthetic(exp):
            paths = get_experiment_output_paths(exp_name, synth_version, batch_mode)
            eval_dir = paths['evaluation']
        else:
            paths = get_real_only_output_paths(exp_name, batch_mode)
            eval_dir = paths['evaluation']

        if not eval_dir.exists():
            print(f"  {exp_name}: No evaluation directory found")
            all_metrics[exp_name] = {'status': 'not_evaluated'}
            continue

        metrics = extract_coco_metrics(eval_dir)

        if not metrics:
            print(f"  {exp_name}: No metrics found")
            all_metrics[exp_name] = {'status': 'no_metrics'}
        else:
            print(f"  {exp_name}: mAP={metrics.get('mAP', 'N/A')}")
            metrics['status'] = 'completed'
            all_metrics[exp_name] = metrics

        # Add experiment metadata
        all_metrics[exp_name].update({
            'exp_id': exp['id'],
            'synth_ratio': exp['synth_ratio'],
            'real_ratio': exp['real_ratio'],
            'strategy': exp['strategy'],
            'description': exp['description'],
            'synth_version': synth_version if experiment_uses_synthetic(exp) else 'N/A',
        })

        # Add training info
        training_info = extract_training_info(exp_name, synth_version, batch_mode)
        if training_info:
            all_metrics[exp_name]['training'] = training_info

        # Add score distribution
        score_dist = extract_score_distribution(exp_name, synth_version, batch_mode)
        if score_dist:
            all_metrics[exp_name]['score_distribution'] = score_dist

        # Add AP by size
        ap_by_size = extract_ap_by_size(exp_name, synth_version, batch_mode)
        if ap_by_size:
            all_metrics[exp_name]['ap_by_size'] = ap_by_size

        # Add FP/FN analysis
        fp_fn = extract_fp_fn_analysis(exp_name, synth_version, batch_mode)
        if fp_fn:
            all_metrics[exp_name]['fp_fn'] = fp_fn

    return all_metrics


# =============================================================================
# TABLE GENERATION
# =============================================================================

def create_comparison_table(metrics: Dict[str, Dict]) -> str:
    """
    Create a comparison table as CSV string.

    Args:
        metrics: Dict mapping experiment name to metrics

    Returns:
        CSV string
    """
    # Define columns
    base_cols = ['exp_id', 'exp_name', 'synth_ratio', 'real_ratio', 'strategy',
                 'mAP', 'mAP_50', 'mAP_75']
    class_cols = [f'AP_{name}' for name in CLASS_NAMES]

    all_cols = base_cols + class_cols

    lines = [','.join(all_cols)]

    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        exp_metrics = metrics.get(exp_name, {})

        row = []
        for col in all_cols:
            if col == 'exp_name':
                row.append(exp_name)
            elif col in exp_metrics:
                val = exp_metrics[col]
                if isinstance(val, float):
                    row.append(f'{val:.4f}')
                else:
                    row.append(str(val))
            else:
                row.append('')

        lines.append(','.join(row))

    return '\n'.join(lines)


def create_pandas_dataframe(metrics: Dict[str, Dict]):
    """Create pandas DataFrame from metrics."""
    if not HAS_PANDAS:
        return None

    rows = []
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        exp_metrics = metrics.get(exp_name, {})

        row = {
            'exp_id': exp['id'],
            'exp_name': exp_name,
            'synth_ratio': exp['synth_ratio'] if isinstance(exp['synth_ratio'], (int, float)) else -1,
            'real_ratio': exp['real_ratio'],
            'strategy': exp['strategy'],
            'description': exp['description'],
        }

        # Add metrics
        for key, value in exp_metrics.items():
            if key not in row:
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_map_vs_real_ratio(df, output_path: Path) -> None:
    """Plot mAP vs percentage of real data comparing synth+real vs real-only."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    # Check if mAP column exists and has valid data
    if 'mAP' not in df.columns or df['mAP'].isna().all():
        print(f"  Skipped {output_path.name}: No mAP data available")
        return

    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 7))

    # Synth+Real experiments (exp1-7): progression from 0% to 100% real WITH 100% synthetic
    synth_real_ids = [1, 2, 3, 4, 5, 6, 7]
    synth_real_df = df[df['exp_id'].isin(synth_real_ids)].copy()
    synth_real_df = synth_real_df.sort_values('real_ratio')

    # Real-only experiments (exp11-14 + exp7): progression with ONLY real data
    real_only_ids = [11, 12, 13, 14]
    real_only_df = df[df['exp_id'].isin(real_only_ids)].copy()
    real_only_df = real_only_df.sort_values('real_ratio')

    # Also add exp7 (100% real, 0% synth) as a reference point for real-only line
    exp7 = df[df['exp_id'] == 7]
    if len(exp7) > 0:
        real_only_df = pd.concat([real_only_df, exp7], ignore_index=True)
        real_only_df = real_only_df.sort_values('real_ratio')

    # Plot Synth+Real line (blue)
    if len(synth_real_df) > 0:
        ax.plot(synth_real_df['real_ratio'] * 100, synth_real_df['mAP'],
                'b-o', linewidth=2.5, markersize=10, label='100% Synth + Real', color='#3498db')
        for _, row in synth_real_df.iterrows():
            if not pd.isna(row['mAP']):
                ax.annotate(f"{row['mAP']:.3f}",
                           (row['real_ratio'] * 100, row['mAP']),
                           textcoords="offset points",
                           xytext=(0, 12),
                           ha='center',
                           fontsize=8,
                           color='#2980b9')

    # Plot Real-only line (green)
    if len(real_only_df) > 0:
        ax.plot(real_only_df['real_ratio'] * 100, real_only_df['mAP'],
                'g-s', linewidth=2.5, markersize=10, label='Real Only', color='#27ae60')
        for _, row in real_only_df.iterrows():
            if not pd.isna(row['mAP']):
                ax.annotate(f"{row['mAP']:.3f}",
                           (row['real_ratio'] * 100, row['mAP']),
                           textcoords="offset points",
                           xytext=(0, -18),
                           ha='center',
                           fontsize=8,
                           color='#1e8449')

    # Add synthetic-only baseline (exp1) as horizontal reference
    exp1 = df[df['exp_id'] == 1]
    if len(exp1) > 0 and not pd.isna(exp1.iloc[0]['mAP']):
        synth_only_map = exp1.iloc[0]['mAP']
        ax.axhline(y=synth_only_map, color='#e74c3c', linestyle='--', linewidth=1.5,
                   label=f'Synth-Only Baseline ({synth_only_map:.3f})', alpha=0.7)

    ax.set_xlabel('Percentage of Real Data (%)', fontsize=12)
    ax.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax.set_title('Model Performance: Synth+Real vs Real-Only Training\n'
                 '(Impact of Adding Synthetic Data at Different Real Data Levels)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-5, 105)
    ax.legend(loc='lower right', fontsize=10)

    # Add annotation for the benefit zone
    if len(synth_real_df) > 0 and len(real_only_df) > 0:
        ax.fill_between(synth_real_df['real_ratio'] * 100, synth_real_df['mAP'],
                        alpha=0.1, color='#3498db')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_minority_class_analysis(df, output_path: Path) -> None:
    """Plot AP for minority classes across experiments."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    exp_names = df['exp_name'].tolist()
    x = range(len(exp_names))
    width = 0.35

    # Get AP for minority classes
    ap_can = df['AP_Can'].fillna(0).tolist() if 'AP_Can' in df.columns else [0] * len(df)
    ap_metal = df['AP_Metal_Debris'].fillna(0).tolist() if 'AP_Metal_Debris' in df.columns else [0] * len(df)

    bars1 = ax.bar([i - width/2 for i in x], ap_can, width, label='Can', color='#2ecc71')
    bars2 = ax.bar([i + width/2 for i in x], ap_metal, width, label='Metal_Debris', color='#e74c3c')

    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('Minority Class Performance Across Experiments', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"exp{df.iloc[i]['exp_id']:02d}" for i in x], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_class_comparison(df, output_path: Path) -> None:
    """Plot per-class AP for selected experiments."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    # Compare key experiments: exp01 (synth), exp07 (real), exp08 (minor), exp09 (prop)
    key_exps = [1, 7, 8, 9]
    plot_df = df[df['exp_id'].isin(key_exps)]

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    x = range(len(CLASS_NAMES))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ap_values = [row.get(f'AP_{name}', 0) or 0 for name in CLASS_NAMES]
        offset = (i - len(plot_df)/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], ap_values, width,
               label=row['exp_name'], color=colors[i % len(colors)])

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('Per-Class AP Comparison Across Key Experiments', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_paired_comparison(df, output_path: Path) -> None:
    """Plot side-by-side comparison of real-only vs synth+real experiments."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    # Comparison pairs: (real_pct, real_only_name, synth_real_name)
    pairs = [
        (10, 'exp11_real10', 'exp02_synth100_real10'),
        (25, 'exp12_real25', 'exp03_synth100_real25'),
        (50, 'exp13_real50', 'exp04_synth100_real50'),
        (75, 'exp14_real75', 'exp05_synth100_real75'),
        (100, 'exp07_real100', 'exp06_synth100_real100'),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    x_labels = []
    real_only_maps = []
    synth_real_maps = []

    for pct, real_name, synth_name in pairs:
        real_row = df[df['exp_name'] == real_name] if 'exp_name' in df.columns else None
        synth_row = df[df['exp_name'] == synth_name] if 'exp_name' in df.columns else None

        real_map = real_row['mAP'].values[0] if real_row is not None and len(real_row) > 0 and 'mAP' in real_row.columns else 0
        synth_map = synth_row['mAP'].values[0] if synth_row is not None and len(synth_row) > 0 and 'mAP' in synth_row.columns else 0

        x_labels.append(f'{pct}%')
        real_only_maps.append(real_map or 0)
        synth_real_maps.append(synth_map or 0)

    x = range(len(x_labels))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], real_only_maps, width,
                   label='Real Only', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], synth_real_maps, width,
                   label='100% Synth + Real', color='#3498db', alpha=0.8)

    # Add value labels and delta annotations
    for i, (r, s) in enumerate(zip(real_only_maps, synth_real_maps)):
        if r > 0:
            ax.annotate(f'{r:.3f}', (i - width/2, r + 0.01), ha='center', fontsize=9)
        if s > 0:
            ax.annotate(f'{s:.3f}', (i + width/2, s + 0.01), ha='center', fontsize=9)
        # Delta annotation
        if r > 0 and s > 0:
            delta = s - r
            color = 'green' if delta > 0 else 'red'
            ax.annotate(f'Δ={delta:+.3f}', (i, max(r, s) + 0.04), ha='center',
                       fontsize=8, color=color, weight='bold')

    ax.set_xlabel('Percentage of Real Data', fontsize=12)
    ax.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax.set_title('Real-Only vs Synthetic+Real Training Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_synth_benefit_curve(df, output_path: Path) -> None:
    """Plot the benefit of adding synthetic data at each real data level."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    pairs = [
        (10, 'exp11_real10', 'exp02_synth100_real10'),
        (25, 'exp12_real25', 'exp03_synth100_real25'),
        (50, 'exp13_real50', 'exp04_synth100_real50'),
        (75, 'exp14_real75', 'exp05_synth100_real75'),
        (100, 'exp07_real100', 'exp06_synth100_real100'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = []
    real_maps = []
    synth_maps = []
    deltas = []

    for pct, real_name, synth_name in pairs:
        real_row = df[df['exp_name'] == real_name] if 'exp_name' in df.columns else None
        synth_row = df[df['exp_name'] == synth_name] if 'exp_name' in df.columns else None

        r = real_row['mAP'].values[0] if real_row is not None and len(real_row) > 0 and 'mAP' in real_row.columns else 0
        s = synth_row['mAP'].values[0] if synth_row is not None and len(synth_row) > 0 and 'mAP' in synth_row.columns else 0

        if r and s:  # Only include if both have values
            x.append(pct)
            real_maps.append(r)
            synth_maps.append(s)
            deltas.append(s - r)

    if not x:
        print(f"  Skipped {output_path.name}: No paired data available")
        plt.close()
        return

    # Left plot: Both curves
    ax1.plot(x, real_maps, 'g-o', label='Real Only', linewidth=2, markersize=8)
    ax1.plot(x, synth_maps, 'b-s', label='Synth + Real', linewidth=2, markersize=8)
    ax1.fill_between(x, real_maps, synth_maps, alpha=0.3,
                     color='blue' if sum(deltas) > 0 else 'red')
    ax1.set_xlabel('% Real Data', fontsize=12)
    ax1.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax1.set_title('Performance Progression', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Right plot: Delta (benefit of synthetic)
    colors = ['green' if d > 0 else 'red' for d in deltas]
    ax2.bar(range(len(x)), deltas, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('% Real Data', fontsize=12)
    ax2.set_ylabel('mAP Improvement from Synthetic', fontsize=12)
    ax2.set_title('Benefit of Adding Synthetic Data', fontsize=14)
    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels([f'{p}%' for p in x])
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def create_paired_comparison_csv(metrics: Dict, output_path: Path) -> None:
    """Create a CSV with paired comparison data."""
    pairs = [
        (10, 'exp11_real10', 'exp02_synth100_real10'),
        (25, 'exp12_real25', 'exp03_synth100_real25'),
        (50, 'exp13_real50', 'exp04_synth100_real50'),
        (75, 'exp14_real75', 'exp05_synth100_real75'),
        (100, 'exp07_real100', 'exp06_synth100_real100'),
    ]

    lines = [
        "real_pct,real_only_exp,real_only_mAP,synth_real_exp,synth_real_mAP,delta,delta_pct"
    ]

    for pct, real_name, synth_name in pairs:
        real_map = metrics.get(real_name, {}).get('mAP', 0) or 0
        synth_map = metrics.get(synth_name, {}).get('mAP', 0) or 0
        delta = synth_map - real_map if real_map and synth_map else 0
        delta_pct = (delta / real_map * 100) if real_map > 0 else 0

        lines.append(f"{pct},{real_name},{real_map:.4f},{synth_name},{synth_map:.4f},{delta:+.4f},{delta_pct:+.1f}%")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {output_path}")


def plot_training_strategy_comparison(df, output_path: Path) -> None:
    """
    Create comprehensive comparison of all training strategies:
    - Synthetic-only (exp01)
    - Real-only (exp11-14, exp07)
    - Synth+Real combined (exp02-06, exp08-09)
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # =========================================================================
    # Plot 1: Strategy Overview (top-left)
    # =========================================================================
    ax1 = axes[0, 0]

    # Get data for each strategy
    synth_only = df[df['exp_id'] == 1]
    real_only = df[df['exp_id'].isin([11, 12, 13, 14, 7])]
    synth_real = df[df['exp_id'].isin([2, 3, 4, 5, 6])]
    special = df[df['exp_id'].isin([8, 9])]

    strategies = ['Synth-Only\n(exp01)', 'Real-Only\n(exp11-14,07)', 'Synth+Real\n(exp02-06)', 'Special\n(exp08-09)']
    best_maps = []
    colors = ['#e74c3c', '#27ae60', '#3498db', '#9b59b6']

    for exp_df in [synth_only, real_only, synth_real, special]:
        if len(exp_df) > 0 and 'mAP' in exp_df.columns:
            valid_maps = exp_df['mAP'].dropna()
            best_maps.append(valid_maps.max() if len(valid_maps) > 0 else 0)
        else:
            best_maps.append(0)

    bars = ax1.bar(strategies, best_maps, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, best_maps):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Best mAP@0.5:0.95', fontsize=12)
    ax1.set_title('Best Performance by Training Strategy', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, max(best_maps) * 1.15 if max(best_maps) > 0 else 1)

    # =========================================================================
    # Plot 2: Real Data Scaling (top-right)
    # =========================================================================
    ax2 = axes[0, 1]

    # Real-only progression
    real_pcts = [10, 25, 50, 75, 100]
    real_exp_ids = [11, 12, 13, 14, 7]

    real_maps = []
    synth_real_maps = []
    synth_exp_ids = [2, 3, 4, 5, 6]

    for exp_id in real_exp_ids:
        exp_row = df[df['exp_id'] == exp_id]
        val = exp_row['mAP'].values[0] if len(exp_row) > 0 and 'mAP' in exp_row.columns else np.nan
        real_maps.append(val if not pd.isna(val) else 0)

    for exp_id in synth_exp_ids:
        exp_row = df[df['exp_id'] == exp_id]
        val = exp_row['mAP'].values[0] if len(exp_row) > 0 and 'mAP' in exp_row.columns else np.nan
        synth_real_maps.append(val if not pd.isna(val) else 0)

    x = np.arange(len(real_pcts))
    width = 0.35

    bars1 = ax2.bar(x - width/2, real_maps, width, label='Real Only', color='#27ae60', alpha=0.8)
    bars2 = ax2.bar(x + width/2, synth_real_maps, width, label='100% Synth + Real', color='#3498db', alpha=0.8)

    # Add delta annotations
    for i, (r, s) in enumerate(zip(real_maps, synth_real_maps)):
        if r > 0 and s > 0:
            delta = s - r
            color = '#27ae60' if delta > 0 else '#e74c3c'
            y_pos = max(r, s) + 0.02
            ax2.annotate(f'Δ={delta:+.2f}', (i, y_pos), ha='center', fontsize=9,
                        color=color, fontweight='bold')

    ax2.set_xlabel('Real Data Percentage', fontsize=12)
    ax2.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax2.set_title('Impact of Synthetic Data at Each Real Data Level', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{p}%' for p in real_pcts])
    ax2.legend(loc='lower right')
    ax2.grid(True, axis='y', alpha=0.3)

    # =========================================================================
    # Plot 3: All Experiments Overview (bottom-left)
    # =========================================================================
    ax3 = axes[1, 0]

    all_exp_ids = sorted(df['exp_id'].unique())
    all_maps = []
    exp_colors = []

    for exp_id in all_exp_ids:
        exp_row = df[df['exp_id'] == exp_id]
        val = exp_row['mAP'].values[0] if len(exp_row) > 0 and 'mAP' in exp_row.columns else 0
        all_maps.append(val if not pd.isna(val) else 0)

        # Color coding
        if exp_id == 1:  # Synth-only
            exp_colors.append('#e74c3c')
        elif exp_id in [11, 12, 13, 14, 7]:  # Real-only
            exp_colors.append('#27ae60')
        elif exp_id in [8, 9]:  # Special strategies
            exp_colors.append('#9b59b6')
        else:  # Synth+Real
            exp_colors.append('#3498db')

    bars = ax3.bar(range(len(all_exp_ids)), all_maps, color=exp_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('Experiment ID', fontsize=12)
    ax3.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax3.set_title('All Experiments Performance Overview', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(all_exp_ids)))
    ax3.set_xticklabels([f'E{eid}' for eid in all_exp_ids], rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Synth-Only'),
        Patch(facecolor='#27ae60', label='Real-Only'),
        Patch(facecolor='#3498db', label='Synth+Real'),
        Patch(facecolor='#9b59b6', label='Special Strategy'),
    ]
    ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # =========================================================================
    # Plot 4: Synthetic Data Benefit Summary (bottom-right)
    # =========================================================================
    ax4 = axes[1, 1]

    # Calculate benefit metrics
    benefit_data = []
    labels = []

    for i, (real_pct, real_id, synth_id) in enumerate([
        (10, 11, 2), (25, 12, 3), (50, 13, 4), (75, 14, 5), (100, 7, 6)
    ]):
        real_exp = df[df['exp_id'] == real_id]
        synth_exp = df[df['exp_id'] == synth_id]

        if len(real_exp) > 0 and len(synth_exp) > 0:
            r = real_exp['mAP'].values[0] if 'mAP' in real_exp.columns else 0
            s = synth_exp['mAP'].values[0] if 'mAP' in synth_exp.columns else 0
            if r and s and not pd.isna(r) and not pd.isna(s):
                benefit_pct = ((s - r) / r) * 100 if r > 0 else 0
                benefit_data.append(benefit_pct)
                labels.append(f'{real_pct}%')

    if benefit_data:
        colors = ['#27ae60' if b > 0 else '#e74c3c' for b in benefit_data]
        bars = ax4.bar(labels, benefit_data, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, benefit_data):
            y_offset = 1 if val >= 0 else -3
            va = 'bottom' if val >= 0 else 'top'
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                    f'{val:+.1f}%', ha='center', va=va, fontsize=10, fontweight='bold')

        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Real Data Percentage', fontsize=12)
        ax4.set_ylabel('mAP Improvement (%)', fontsize=12)
        ax4.set_title('Relative Benefit of Adding Synthetic Data', fontsize=14, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Comprehensive Training Strategy Analysis\n', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_class_strategy_comparison(df, output_path: Path) -> None:
    """
    Compare per-class AP for key strategies:
    - Best synth-only
    - Best real-only
    - Best synth+real
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    import numpy as np

    fig, ax = plt.subplots(figsize=(14, 8))

    # Find best experiment from each strategy
    synth_only = df[df['exp_id'] == 1]
    real_only = df[df['exp_id'].isin([11, 12, 13, 14, 7])]
    synth_real = df[df['exp_id'].isin([2, 3, 4, 5, 6, 8, 9])]

    def get_best_exp(exp_df):
        if len(exp_df) == 0 or 'mAP' not in exp_df.columns:
            return None
        valid = exp_df.dropna(subset=['mAP'])
        if len(valid) == 0:
            return None
        return valid.loc[valid['mAP'].idxmax()]

    best_synth = synth_only.iloc[0] if len(synth_only) > 0 else None
    best_real = get_best_exp(real_only)
    best_combined = get_best_exp(synth_real)

    strategies = []
    strategy_data = []
    colors = []

    if best_synth is not None:
        strategies.append(f"Synth-Only\n({best_synth['exp_name']})")
        strategy_data.append(best_synth)
        colors.append('#e74c3c')

    if best_real is not None:
        strategies.append(f"Best Real-Only\n({best_real['exp_name']})")
        strategy_data.append(best_real)
        colors.append('#27ae60')

    if best_combined is not None:
        strategies.append(f"Best Synth+Real\n({best_combined['exp_name']})")
        strategy_data.append(best_combined)
        colors.append('#3498db')

    if not strategy_data:
        print(f"  Skipped {output_path.name}: No data available")
        plt.close()
        return

    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(strategy_data)

    for i, (exp_row, color, label) in enumerate(zip(strategy_data, colors, strategies)):
        ap_values = []
        for class_name in CLASS_NAMES:
            val = exp_row.get(f'AP_{class_name}', 0)
            ap_values.append(val if val and not pd.isna(val) else 0)

        offset = (i - len(strategy_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, ap_values, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('Per-Class AP Comparison: Best Model from Each Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    # Highlight minority classes
    for i, class_name in enumerate(CLASS_NAMES):
        if class_name in MINORITY_CLASSES:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='yellow')
            ax.text(i, ax.get_ylim()[1] * 0.95, '⚠️', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def _copy_evaluation_images(
    output_dir: Path,
    exp_list: List[Dict[str, Any]],
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, Dict[str, Path]]:
    """
    Copy ALL relevant images from evaluation directories to results folder.

    Args:
        output_dir: Target directory for copied images
        exp_list: List of experiment config dicts (from ALL_EXPERIMENTS)
        synth_version: Synthetic dataset version for versioned paths
        batch_mode: Batch sampling mode

    Returns:
        Dict mapping exp_name to dict of image type -> relative path
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Images to copy from evaluation directory
    eval_image_files = [
        "confusion_matrix_normalized.png",
        "confusion_matrix.png",
        "per_class_ap.png",
        "score_distributions.png",
        "fp_fn_bar_chart.png",
        "ap_ar_by_size.png",
        "classification_report.png",
        "dataset_distribution.png",
    ]

    # Images to copy from training directory
    train_image_files = [
        "train_dataset_analysis.png",
        "val_dataset_analysis.png",
    ]

    copied_images = {}
    stats = {'total_experiments': 0, 'with_images': 0, 'total_images': 0, 'missing_dirs': []}

    for exp in exp_list:
        exp_name = exp['name']
        copied_images[exp_name] = {}
        stats['total_experiments'] += 1

        # Get versioned paths based on experiment type
        if experiment_uses_synthetic(exp):
            paths = get_experiment_output_paths(exp_name, synth_version, batch_mode)
        else:
            paths = get_real_only_output_paths(exp_name, batch_mode)

        # Create subfolder for each experiment
        exp_img_dir = images_dir / exp_name
        exp_img_dir.mkdir(parents=True, exist_ok=True)

        exp_images_found = 0

        # Copy evaluation images
        eval_dir = paths['evaluation'] / "best"
        if eval_dir.exists():
            for img_file in eval_image_files:
                src = eval_dir / img_file
                if src.exists():
                    dst = exp_img_dir / img_file
                    try:
                        shutil.copy2(src, dst)
                        copied_images[exp_name][img_file.replace('.png', '')] = f"images/{exp_name}/{img_file}"
                        exp_images_found += 1
                    except Exception as e:
                        print(f"    Warning: Could not copy {src}: {e}")
        else:
            stats['missing_dirs'].append(f"{exp_name}: eval dir not found ({eval_dir})")

        # Copy training images
        train_dir = paths['training']
        if train_dir.exists():
            for img_file in train_image_files:
                src = train_dir / img_file
                if src.exists():
                    dst = exp_img_dir / img_file
                    try:
                        shutil.copy2(src, dst)
                        copied_images[exp_name][img_file.replace('.png', '')] = f"images/{exp_name}/{img_file}"
                        exp_images_found += 1
                    except Exception as e:
                        print(f"    Warning: Could not copy {src}: {e}")

        if exp_images_found > 0:
            stats['with_images'] += 1
            stats['total_images'] += exp_images_found
            print(f"    {exp_name}: copied {exp_images_found} images")

    # Print summary
    print(f"  Image copy summary:")
    print(f"    - Experiments with images: {stats['with_images']}/{stats['total_experiments']}")
    print(f"    - Total images copied: {stats['total_images']}")
    if stats['missing_dirs']:
        print(f"    - Experiments with missing directories: {len(stats['missing_dirs'])}")
        for msg in stats['missing_dirs'][:5]:  # Show first 5
            print(f"      • {msg}")
        if len(stats['missing_dirs']) > 5:
            print(f"      ... and {len(stats['missing_dirs']) - 5} more")

    return copied_images


def _extract_diagnostic_summary(
    exp: Dict[str, Any],
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Optional[Dict[str, Any]]:
    """Extract diagnostic summary data from evaluation output.

    Args:
        exp: Experiment configuration dict
        synth_version: Synthetic dataset version for versioned paths
        batch_mode: Batch sampling mode
    """
    exp_name = exp['name']

    # Get versioned paths based on experiment type
    if experiment_uses_synthetic(exp):
        paths = get_experiment_output_paths(exp_name, synth_version, batch_mode)
    else:
        paths = get_real_only_output_paths(exp_name, batch_mode)

    eval_dir = paths['evaluation'] / "best"
    diag_file = eval_dir / "diagnostic_summary.txt"

    if not diag_file.exists():
        return None

    try:
        with open(diag_file, 'r', encoding='utf-8') as f:
            content = f.read()

        summary = {}

        # Extract global stats
        patterns = {
            'total_gt': r'Total Ground Truth:\s+(\d+)',
            'total_pred': r'Total Predicciones:\s+(\d+)',
            'true_positives': r'True Positives \(TP\):\s+(\d+)',
            'false_negatives': r'False Negatives \(FN\):\s+(\d+)',
            'false_positives': r'False Positives \(FP\):\s+(\d+)',
            'micro_precision': r'Micro-Precision:\s+([\d.]+)%',
            'micro_recall': r'Micro-Recall:\s+([\d.]+)%',
            'micro_f1': r'Micro-F1:\s+([\d.]+)',
            'localization_recall': r'Localization Recall:\s+([\d.]+)%',
            'mean_iou': r'Mean IoU.*:\s+([\d.]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                val = match.group(1)
                summary[key] = float(val) if '.' in val else int(val)

        # Extract per-class stats
        class_pattern = r'(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)'
        summary['per_class'] = {}
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            if class_name in CLASS_NAMES:
                summary['per_class'][class_name] = {
                    'gt': int(match.group(2)),
                    'tp': int(match.group(3)),
                    'fn': int(match.group(4)),
                    'fp': int(match.group(5)),
                    'recall': float(match.group(6)),
                    'precision': float(match.group(7)),
                    'f1': float(match.group(8)),
                }

        return summary
    except Exception as e:
        print(f"  Warning: Could not parse diagnostic summary for {exp_name}: {e}")
        return None


def _get_exp_data(df, exp_id: int):
    """Helper to get experiment data by ID."""
    if df is None or 'exp_id' not in df.columns:
        return None
    matches = df[df['exp_id'] == exp_id]
    return matches.iloc[0] if len(matches) > 0 else None


def _calc_improvement(baseline: float, new_value: float) -> str:
    """Calculate percentage improvement."""
    if baseline == 0:
        return "N/A"
    pct = ((new_value - baseline) / baseline) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def generate_markdown_report(
    df,
    metrics: Dict,
    output_path: Path,
    lang: str = 'en',
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> None:
    """Generate a comprehensive markdown report with automatic analysis and images.

    Args:
        df: Pandas DataFrame with experiment metrics
        metrics: Dict mapping experiment name to comprehensive metrics
        output_path: Path where to save the report
        lang: Language code ('en' for English, 'es' for Spanish)
        synth_version: Synthetic dataset version for versioned paths
        batch_mode: Batch sampling mode
    """

    output_dir = output_path.parent

    # Copy ALL evaluation images to results directory (including real-only experiments)
    print("  Copying evaluation and training images...")
    copied_images = _copy_evaluation_images(output_dir, ALL_EXPERIMENTS, synth_version, batch_mode)

    # Extract diagnostic summaries for ALL experiments (including real-only)
    print("  Extracting diagnostic summaries...")
    diagnostics = {}
    for exp in ALL_EXPERIMENTS:
        diag = _extract_diagnostic_summary(exp, synth_version, batch_mode)
        if diag:
            diagnostics[exp['name']] = diag

    # Check if we have valid data
    has_data = HAS_PANDAS and df is not None and 'mAP' in df.columns and not df['mAP'].isna().all()

    # Get key experiments for comparison
    exp01, exp02, exp06, exp07, exp08, exp09 = None, None, None, None, None, None
    exp11, exp12, exp13, exp14 = None, None, None, None  # Real-only experiments
    best_exp, synth_only_map, real_only_map, best_map = None, 0, 0, 0
    best_real_only_exp = None

    if has_data:
        exp01 = _get_exp_data(df, 1)
        exp02 = _get_exp_data(df, 2)
        exp06 = _get_exp_data(df, 6)
        exp07 = _get_exp_data(df, 7)
        exp08 = _get_exp_data(df, 8)
        exp09 = _get_exp_data(df, 9)
        exp11 = _get_exp_data(df, 11)
        exp12 = _get_exp_data(df, 12)
        exp13 = _get_exp_data(df, 13)
        exp14 = _get_exp_data(df, 14)

        best_idx = df['mAP'].idxmax()
        best_exp = df.loc[best_idx]
        synth_only_map = exp01['mAP'] if exp01 is not None else 0

        # Find best real-only experiment (exp11-14) as the true baseline
        real_only_exps = [exp11, exp12, exp13, exp14]
        real_only_maps = []
        for ro_exp in real_only_exps:
            if ro_exp is not None and 'mAP' in ro_exp and not (isinstance(ro_exp['mAP'], float) and ro_exp['mAP'] != ro_exp['mAP']):
                real_only_maps.append((ro_exp, ro_exp['mAP']))

        if real_only_maps:
            best_real_only_exp, real_only_map = max(real_only_maps, key=lambda x: x[1])
        else:
            # Fallback to exp07 if no real-only experiments
            real_only_map = exp07['mAP'] if exp07 is not None else 0
            best_real_only_exp = exp07

        best_map = best_exp['mAP']

    # Calculate total training time
    total_training_hours = 0
    for exp_name, m in metrics.items():
        if 'training' in m and 'duration_seconds' in m['training']:
            total_training_hours += m['training']['duration_seconds'] / 3600

    # =========================================================================
    # START REPORT
    # =========================================================================
    report = f"""# {t('title', lang)}
## {t('subtitle', lang)}

**{t('generated', lang)}:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## {t('toc', lang)}

1. [{t('executive_summary', lang)}](#executive-summary)
2. [{t('methodology', lang)}](#methodology)
3. [{t('training_overview', lang)}](#training-overview)
4. [{t('key_visualizations', lang)}](#key-visualizations)
5. [{t('complete_results', lang)}](#complete-results-table)
6. [{t('detailed_analysis', lang)}](#detailed-experiment-analysis)
7. [{t('per_class_analysis', lang)}](#per-class-performance-analysis)
8. [{t('error_analysis', lang)}](#error-analysis)
9. [{t('size_analysis', lang)}](#object-size-analysis)
10. [{t('conclusions', lang)}](#conclusions)
11. [{t('recommendations', lang)}](#recommendations)
12. [{t('appendix_cm', lang)}](#appendix-all-confusion-matrices)

---

## {t('executive_summary', lang)}

This report presents a comprehensive analysis of **{len(ALL_EXPERIMENTS)} experiments** ({len(EXPERIMENTS)} synthetic+real + {len(REAL_ONLY_EXPERIMENTS)} real-only baselines) evaluating the impact of synthetic-to-real data ratios on RT-DETRv2 object detection performance for marine debris.

"""

    # Determine baseline experiment name
    baseline_exp_name = best_real_only_exp['exp_name'] if best_real_only_exp is not None else 'N/A'

    if has_data:
        report += f"""### {t('key_results', lang)}

| {t('metric', lang)} | {t('value', lang)} |
|--------|-------|
| **{t('best_model', lang)}** | {best_exp['exp_name']} |
| **Best mAP@0.5:0.95** | **{best_map:.4f}** |
| **Best mAP@0.50** | {best_exp.get('mAP_50', 0):.4f} |
| **{t('real_only_baseline', lang)} ({baseline_exp_name})** | {real_only_map:.4f} mAP |
| **{t('synth_only_baseline', lang)} (exp01)** | {synth_only_map:.4f} mAP |
| **{t('improvement_real', lang)}** | {_calc_improvement(real_only_map, best_map)} |
| **{t('improvement_synth', lang)}** | {_calc_improvement(synth_only_map, best_map)} |
| **{t('total_training_time', lang)}** | {total_training_hours:.1f} {t('hours', lang)} |
| **{t('num_experiments', lang)}** | {len(ALL_EXPERIMENTS)} |
| **{t('object_classes', lang)}** | 8 |

### {t('critical_findings', lang)}

"""
        # Compute domain gap
        if synth_only_map > 0 and real_only_map > 0:
            domain_gap_pct = ((real_only_map - synth_only_map) / real_only_map) * 100
            report += f"""1. **{t('domain_gap', lang)}**: {t('synth_achieves', lang)} **{(synth_only_map/real_only_map)*100:.1f}%**
   {t('of_real_performance', lang)}, {t('performance_gap', lang)}: **{domain_gap_pct:.1f}%**.

"""
        if exp02 is not None and exp01 is not None:
            first_jump = exp02['mAP'] - exp01['mAP']
            report += f"""2. **{t('real_data_value', lang)}**: {t('first_10_pct', lang)} **{first_jump:.4f}**
   ({(first_jump/synth_only_map)*100:.0f}% {t('relative_improvement', lang)}).

"""
        if exp08 is not None and exp07 is not None:
            can_07 = exp07.get('AP_Can', 0) or 0
            can_08 = exp08.get('AP_Can', 0) or 0
            metal_09 = exp09.get('AP_Metal_Debris', 0) or 0 if exp09 is not None else 0
            metal_07 = exp07.get('AP_Metal_Debris', 0) or 0
            report += f"""3. **{t('minority_strategy', lang)}**: {t('targeted_synth', lang)}:
   - Can: {_calc_improvement(can_07, can_08)} (exp08 vs exp07)
   - Metal_Debris: {_calc_improvement(metal_07, metal_09)} (exp09 vs exp07)

"""

    # =========================================================================
    # METHODOLOGY
    # =========================================================================
    report += f"""---

## {t('methodology', lang)}

### {t('model_architecture', lang)}

- **Model**: RT-DETRv2 with ResNet50 backbone ({t('pretrained', lang)})
- **{t('input_resolution', lang)}**: 640x640 pixels
- **{t('loss_functions', lang)}**: Focal Loss + VFL (Varifocal Loss)
- **{t('optimizer', lang)}**: AdamW (LR: 5e-5, Backbone LR: 5e-6)

### {t('experiment_design', lang)}

| Exp ID | Name | Synthetic | Real | Strategy |
|--------|------|-----------|------|----------|
"""
    for exp in ALL_EXPERIMENTS:
        report += f"| {exp['id']} | {exp['name']} | {exp['synth_ratio']} | {exp['real_ratio']} | {exp['strategy']} |\n"

    report += f"""
### {t('dataset_info', lang)}

**{t('real_dataset_dist', lang)}:**

| {t('class', lang)} | {t('samples', lang)} | {t('percentage', lang)} |
|-------|---------|------------|
"""
    total_real = sum(REAL_CLASS_COUNTS.values())
    for class_name in CLASS_NAMES:
        count = REAL_CLASS_COUNTS.get(class_name, 0)
        pct = (count / total_real) * 100 if total_real > 0 else 0
        minority_marker = " ⚠️" if class_name in MINORITY_CLASSES else ""
        report += f"| {class_name}{minority_marker} | {count:,} | {pct:.1f}% |\n"
    report += f"| **{t('total', lang)}** | **{total_real:,}** | **100%** |\n"

    report += f"""
*⚠️ {t('minority_indicator', lang)}*

### {t('training_config', lang)}

- **{t('max_epochs', lang)}**: 200
- **{t('early_stopping', lang)}**: {t('patience_epochs', lang).format(n=20)}, {t('monitoring', lang)} `coco_eval_bbox`
- **{t('batch_size', lang)}**: 32
- **{t('eval_metric', lang)}**: COCO mAP@0.5:0.95

"""

    # =========================================================================
    # TRAINING OVERVIEW
    # =========================================================================
    report += f"""---

## {t('training_overview', lang)}

### {t('training_duration', lang)}

| {t('experiment', lang)} | {t('duration', lang)} | {t('status', lang)} | {t('epochs', lang)} |
|------------|----------|--------|--------|
"""
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        m = metrics.get(exp_name, {})
        train_info = m.get('training', {})
        duration = train_info.get('duration_formatted', 'N/A')
        status = train_info.get('status', 'unknown')
        epochs = train_info.get('total_epochs', 'N/A')
        report += f"| {exp_name} | {duration} | {status} | {epochs} |\n"

    report += f"""
**{t('total_training_time', lang)}**: {total_training_hours:.1f} {t('hours', lang)} ({total_training_hours/24:.1f} {t('days', lang)})

"""

    # =========================================================================
    # KEY VISUALIZATIONS
    # =========================================================================
    report += f"""---

## {t('key_visualizations', lang)}

### 1. {t('map_vs_real', lang)}

{t('map_vs_real_desc', lang)}

![mAP vs Real Data Ratio](mAP_vs_real_ratio.png)

**{t('key_observations', lang)}:**
- {t('sharp_improvement', lang)}
- {t('diminishing_after_50', lang)}
- {t('real_only_underperform', lang)}

### 2. {t('minority_comparison', lang)}

{t('minority_desc', lang)}

![Minority Class Analysis](minority_class_analysis.png)

**{t('key_observations', lang)}:**
- {t('both_benefit', lang)}
- {t('exp08_best_can', lang)}
- {t('exp09_best_metal', lang)}

### 3. {t('per_class_comparison', lang)}

{t('per_class_desc', lang)}

![Per-Class Comparison](per_class_comparison.png)

### 4. Training Strategy Comparison

Comprehensive comparison of all training strategies showing best performance from:
- **Synthetic-Only**: Training with only synthetic data
- **Real-Only**: Training with only real data at various percentages
- **Synth+Real**: Training with combined synthetic and real data
- **Special Strategies**: Minority-focused and proportional sampling

![Training Strategy Comparison](training_strategy_comparison.png)

### 5. Paired Comparison: Real-Only vs Synth+Real

Direct comparison at each real data percentage showing the benefit/detriment of adding synthetic data.

![Paired Comparison](paired_comparison.png)

![Synth Benefit Analysis](synth_benefit_analysis.png)

### 6. Per-Class Strategy Comparison

Comparison of per-class AP between the best model from each training strategy.

![Per-Class Strategy Comparison](per_class_strategy_comparison.png)

"""

    # =========================================================================
    # STRATEGY COMPARISON SECTION
    # =========================================================================
    report += """---

## Training Strategy Comparison Summary

This section provides a direct comparison of the three main training strategies.

### Strategy Overview

"""
    # Calculate strategy summaries
    synth_only_exp = df[df['exp_id'] == 1] if df is not None and 'exp_id' in df.columns else None
    real_only_exps = df[df['exp_id'].isin([11, 12, 13, 14, 7])] if df is not None and 'exp_id' in df.columns else None
    synth_real_exps = df[df['exp_id'].isin([2, 3, 4, 5, 6, 8, 9])] if df is not None and 'exp_id' in df.columns else None

    def get_strategy_stats(exp_df):
        if exp_df is None or len(exp_df) == 0 or 'mAP' not in exp_df.columns:
            return {'count': 0, 'best': 0, 'avg': 0, 'best_exp': 'N/A'}
        valid = exp_df.dropna(subset=['mAP'])
        if len(valid) == 0:
            return {'count': 0, 'best': 0, 'avg': 0, 'best_exp': 'N/A'}
        best_idx = valid['mAP'].idxmax()
        return {
            'count': len(valid),
            'best': valid['mAP'].max(),
            'avg': valid['mAP'].mean(),
            'best_exp': valid.loc[best_idx, 'exp_name'] if 'exp_name' in valid.columns else 'N/A'
        }

    synth_stats = get_strategy_stats(synth_only_exp)
    real_stats = get_strategy_stats(real_only_exps)
    combined_stats = get_strategy_stats(synth_real_exps)

    report += f"""| Strategy | # Experiments | Best mAP | Average mAP | Best Experiment |
|----------|--------------|----------|-------------|-----------------|
| **Synthetic-Only** | {synth_stats['count']} | {synth_stats['best']:.4f} | {synth_stats['avg']:.4f} | {synth_stats['best_exp']} |
| **Real-Only** | {real_stats['count']} | {real_stats['best']:.4f} | {real_stats['avg']:.4f} | {real_stats['best_exp']} |
| **Synth+Real Combined** | {combined_stats['count']} | {combined_stats['best']:.4f} | {combined_stats['avg']:.4f} | {combined_stats['best_exp']} |

"""

    # Calculate improvement percentages
    if synth_stats['best'] > 0:
        real_vs_synth = ((real_stats['best'] - synth_stats['best']) / synth_stats['best']) * 100
        combined_vs_synth = ((combined_stats['best'] - synth_stats['best']) / synth_stats['best']) * 100
        report += f"""### Key Findings

1. **Domain Gap Impact**: Real-only training achieves **{real_vs_synth:+.1f}%** improvement over synthetic-only
2. **Combined Strategy Benefit**: Synth+Real achieves **{combined_vs_synth:+.1f}%** improvement over synthetic-only
"""
        if real_stats['best'] > 0:
            combined_vs_real = ((combined_stats['best'] - real_stats['best']) / real_stats['best']) * 100
            report += f"""3. **Synthetic Data Value**: Adding synthetic data to real provides **{combined_vs_real:+.1f}%** {'improvement' if combined_vs_real > 0 else 'change'} over real-only

"""
        else:
            report += "\n"

    # Paired comparison table
    report += """### Detailed Paired Comparison

| Real % | Real-Only Exp | Real-Only mAP | Synth+Real Exp | Synth+Real mAP | Δ mAP | Δ % |
|--------|--------------|---------------|----------------|----------------|-------|-----|
"""
    pairs = [
        (10, 'exp11_real10', 'exp02_synth100_real10'),
        (25, 'exp12_real25', 'exp03_synth100_real25'),
        (50, 'exp13_real50', 'exp04_synth100_real50'),
        (75, 'exp14_real75', 'exp05_synth100_real75'),
        (100, 'exp07_real100', 'exp06_synth100_real100'),
    ]

    for pct, real_name, synth_name in pairs:
        real_map = metrics.get(real_name, {}).get('mAP', 0) or 0
        synth_map = metrics.get(synth_name, {}).get('mAP', 0) or 0
        delta = synth_map - real_map
        delta_pct = ((delta / real_map) * 100) if real_map > 0 else 0
        delta_sign = '+' if delta > 0 else ''
        report += f"| {pct}% | {real_name} | {real_map:.4f} | {synth_name} | {synth_map:.4f} | {delta_sign}{delta:.4f} | {delta_sign}{delta_pct:.1f}% |\n"

    report += f"""
### Interpretation

- **Low Real Data (10-25%)**: Synthetic data provides the most significant benefit when real data is limited
- **Medium Real Data (50%)**: The benefit of synthetic data starts to diminish
- **High Real Data (75-100%)**: Synthetic data may provide marginal benefit or slight degradation

"""

    # =========================================================================
    # COMPLETE RESULTS TABLE
    # =========================================================================
    report += f"""---

## {t('complete_results', lang)}

### {t('primary_metrics', lang)}

| Exp | Name | mAP | mAP@50 | mAP@75 | AP_small | AP_medium | AP_large |
|-----|------|-----|--------|--------|----------|-----------|----------|
"""
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        m = metrics.get(exp_name, {})
        row = f"| {exp['id']} | {exp_name} "
        for metric in ['mAP', 'mAP_50', 'mAP_75', 'AP_small', 'AP_medium', 'AP_large']:
            val = m.get(metric)
            row += f"| {val:.4f} " if val else "| - "
        report += row + "|\n"

    report += f"""
### {t('per_class_ap', lang)}

| Exp | Bottle | Can | Fish_Net | Glove | Mask | Metal | Plastic | Tire |
|-----|--------|-----|----------|-------|------|-------|---------|------|
"""
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        m = metrics.get(exp_name, {})
        row = f"| {exp['id']} "
        for class_name in CLASS_NAMES:
            val = m.get(f'AP_{class_name}')
            row += f"| {val:.3f} " if val else "| - "
        report += row + "|\n"

    # =========================================================================
    # DETAILED EXPERIMENT ANALYSIS
    # =========================================================================
    report += f"""
---

## {t('detailed_analysis', lang)}

"""

    # Analyze ALL experiments, not just key ones
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        exp_id = exp['id']
        m = metrics.get(exp_name, {})

        # Determine experiment type for title
        if exp['synth_ratio'] == 1.0 and exp['real_ratio'] == 0.0:
            exp_type = f"🔵 {t('synthetic_only', lang)}"
        elif exp['synth_ratio'] == 0.0 and exp['real_ratio'] == 1.0:
            exp_type = f"🟢 {t('real_only', lang)}"
        elif exp['strategy'] == 'minority_only':
            exp_type = f"🟡 {t('minority_focused', lang)}"
        elif exp['strategy'] == 'inverse_proportional':
            exp_type = f"🟣 {t('proportional', lang)}"
        else:
            exp_type = f"🔷 {t('mixed', lang)}"

        report += f"""### {t('experiment', lang)} {exp_id}: {exp_name}

**{t('type', lang)}:** {exp_type}
**{t('description', lang)}:** {exp['description']}

"""
        # Training info
        train_info = m.get('training', {})
        if train_info:
            report += f"""**{t('training_info', lang)}:**
- {t('duration', lang)}: {train_info.get('duration_formatted', 'N/A')}
- {t('epochs', lang)}: {train_info.get('total_epochs', 'N/A')}
- {t('final_loss', lang)}: {train_info.get('final_loss', 'N/A')}

"""

        # Key metrics
        report += f"""**{t('eval_metrics', lang)}:**

| {t('metric', lang)} | {t('value', lang)} |
|--------|-------|
| mAP@0.5:0.95 | {m.get('mAP', 'N/A')} |
| mAP@0.50 | {m.get('mAP_50', 'N/A')} |
| mAP@0.75 | {m.get('mAP_75', 'N/A')} |

"""
        # Diagnostic summary if available
        if exp_name in diagnostics:
            diag = diagnostics[exp_name]
            report += f"""**{t('detection_stats', lang)}:**

| {t('metric', lang)} | {t('value', lang)} |
|--------|-------|
| {t('total_gt', lang)} | {diag.get('total_gt', 'N/A')} |
| {t('total_pred', lang)} | {diag.get('total_pred', 'N/A')} |
| {t('true_positives', lang)} | {diag.get('true_positives', 'N/A')} |
| {t('false_negatives', lang)} | {diag.get('false_negatives', 'N/A')} |
| {t('false_positives', lang)} | {diag.get('false_positives', 'N/A')} |
| Micro-Precision | {diag.get('micro_precision', 'N/A')}% |
| Micro-Recall | {diag.get('micro_recall', 'N/A')}% |
| Micro-F1 | {diag.get('micro_f1', 'N/A')} |

"""
            # Per-class performance
            if 'per_class' in diag and diag['per_class']:
                report += f"""**{t('per_class_detection', lang)}:**

| {t('class', lang)} | GT | TP | FN | FP | Recall | Precision | F1 |
|-------|----|----|----|----|--------|-----------|-----|
"""
                for class_name in CLASS_NAMES:
                    if class_name in diag['per_class']:
                        c = diag['per_class'][class_name]
                        report += f"| {class_name} | {c['gt']} | {c['tp']} | {c['fn']} | {c['fp']} | {c['recall']:.1f}% | {c['precision']:.1f}% | {c['f1']:.3f} |\n"
                report += "\n"

        # Add images
        if exp_name in copied_images:
            imgs = copied_images[exp_name]
            if 'confusion_matrix_normalized' in imgs:
                report += f"""<details>
<summary>📊 {t('click_expand_cm', lang)}</summary>

![Confusion Matrix - {exp_name}]({imgs['confusion_matrix_normalized']})

</details>

"""
            if 'per_class_ap' in imgs:
                report += f"""<details>
<summary>📈 {t('click_expand_ap', lang)}</summary>

![Per-Class AP - {exp_name}]({imgs['per_class_ap']})

</details>

"""
            if 'score_distributions' in imgs:
                report += f"""<details>
<summary>📉 {t('click_expand_scores', lang)}</summary>

![Score Distributions - {exp_name}]({imgs['score_distributions']})

</details>

"""

        report += "---\n\n"

    # =========================================================================
    # PER-CLASS PERFORMANCE ANALYSIS
    # =========================================================================
    report += f"""## {t('per_class_analysis', lang)}

### {t('class_comparison', lang)}

"""
    for class_name in CLASS_NAMES:
        minority_tag = f" (⚠️ {t('minority_class', lang)})" if class_name in MINORITY_CLASSES else ""
        report += f"""#### {class_name}{minority_tag}

| {t('experiment', lang)} | AP@0.5:0.95 | AP@0.50 | AP@0.75 |
|------------|-------------|---------|---------|
"""
        for exp in ALL_EXPERIMENTS:
            exp_name = exp['name']
            m = metrics.get(exp_name, {})
            ap = m.get(f'AP_{class_name}', '-')
            ap50 = m.get(f'AP50_{class_name}', '-')
            ap75 = m.get(f'AP75_{class_name}', '-')
            ap_str = f"{ap:.4f}" if isinstance(ap, float) else ap
            ap50_str = f"{ap50:.4f}" if isinstance(ap50, float) else ap50
            ap75_str = f"{ap75:.4f}" if isinstance(ap75, float) else ap75
            report += f"| {exp_name} | {ap_str} | {ap50_str} | {ap75_str} |\n"
        report += "\n"

    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================
    report += f"""---

## {t('error_analysis', lang)}

### {t('fp_fn_analysis', lang)}

{t('fp_fn_desc', lang)}

"""
    # Compare FP/FN for key experiments
    key_exp_names = ['exp01_synth100', 'exp07_real100', 'exp09_real_synth_prop']
    for exp_name in key_exp_names:
        m = metrics.get(exp_name, {})
        fp_fn = m.get('fp_fn', {})
        if fp_fn:
            report += f"""#### {exp_name}

| {t('class', lang)} | {t('false_positives', lang)} | {t('false_negatives', lang)} | FP/(FP+FN) Ratio |
|-------|-----------------|-----------------|------------------|
"""
            for class_name in CLASS_NAMES:
                if class_name in fp_fn:
                    fp = fp_fn[class_name]['fp']
                    fn = fp_fn[class_name]['fn']
                    ratio = fp / (fp + fn) if (fp + fn) > 0 else 0
                    report += f"| {class_name} | {fp} | {fn} | {ratio:.2%} |\n"
            report += "\n"

    # =========================================================================
    # OBJECT SIZE ANALYSIS
    # =========================================================================
    report += f"""---

## {t('size_analysis', lang)}

{t('size_variation', lang)} (small < 32², medium 32²-96², large > 96²).

| {t('experiment', lang)} | AP_small | AP_medium | AP_large | AR_small | AR_medium | AR_large |
|------------|----------|-----------|----------|----------|-----------|----------|
"""
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        m = metrics.get(exp_name, {})
        size_data = m.get('ap_by_size', {})
        if size_data:
            report += f"| {exp_name} "
            for metric in ['AP_small', 'AP_medium', 'AP_large', 'AR_small', 'AR_medium', 'AR_large']:
                val = size_data.get(metric)
                report += f"| {val:.4f} " if val else "| - "
            report += "|\n"
        else:
            ap_small = m.get('AP_small')
            ap_medium = m.get('AP_medium')
            ap_large = m.get('AP_large')
            report += f"| {exp_name} "
            report += f"| {ap_small:.4f} " if ap_small else "| - "
            report += f"| {ap_medium:.4f} " if ap_medium else "| - "
            report += f"| {ap_large:.4f} " if ap_large else "| - "
            report += "| - | - | - |\n"

    report += f"""
**{t('key_observations', lang)}:**
- {t('size_obs_small', lang)}
- {t('size_obs_large', lang)}
- {t('size_obs_medium', lang)}

"""

    # =========================================================================
    # CONCLUSIONS
    # =========================================================================
    report += f"""---

## {t('conclusions', lang)}

### {t('hypothesis_validation', lang)}

"""

    if has_data:
        # 1. Synthetic-only performance
        report += f"""#### 1. {t('synth_domain_gap', lang)}

"""
        if exp01 is not None and exp07 is not None:
            synth_gap = real_only_map - synth_only_map
            synth_pct = (synth_only_map / real_only_map * 100) if real_only_map > 0 else 0
            report += f"""| {t('metric', lang)} | {t('value', lang)} |
|--------|-------|
| Synthetic-only mAP | {synth_only_map:.4f} |
| Real-only mAP | {real_only_map:.4f} |
| {t('performance_gap', lang)} | {synth_gap:.4f} ({100-synth_pct:.1f}% loss) |
| {t('synth_achieves', lang)} | {synth_pct:.1f}% {t('of_real_performance', lang)} |

**{t('conclusion', lang)}**: {t('domain_gap_conclusion', lang)} **{100-synth_pct:.1f}%**
{t('performance_degradation', lang)} **{t('insufficient', lang)}** {t('for_production', lang)}

"""

        # 2. Progressive improvement
        report += f"""#### 2. {t('real_data_impact', lang)}

"""
        progression_exps = [1, 2, 3, 4, 5, 6]
        progression_data = []
        for eid in progression_exps:
            exp_data = _get_exp_data(df, eid)
            if exp_data is not None:
                progression_data.append({
                    'id': eid,
                    'real_pct': exp_data['real_ratio'] * 100,
                    'mAP': exp_data['mAP']
                })

        if len(progression_data) >= 2:
            report += f"| {t('real_data_pct', lang)} | mAP | {t('delta_map', lang)} | {t('relative_improvement', lang)} |\n"
            report += "|-------------|-----|-------|---------------------|\n"
            base_map = progression_data[0]['mAP']
            prev_map = None
            for p in progression_data:
                delta = p['mAP'] - base_map
                imp = _calc_improvement(prev_map, p['mAP']) if prev_map else "-"
                report += f"| {p['real_pct']:.0f}% | {p['mAP']:.4f} | +{delta:.4f} | {imp} |\n"
                prev_map = p['mAP']

            if exp01 is not None and exp02 is not None:
                first_jump = exp02['mAP'] - exp01['mAP']
                first_jump_pct = (first_jump / exp01['mAP'] * 100) if exp01['mAP'] > 0 else 0
                report += f"""
**{t('key_findings', lang)}:**
- {t('first_10_pct', lang)} **{first_jump_pct:.0f}%** {t('relative_improvement', lang)} (+{first_jump:.4f} mAP)
- {t('diminishing_returns', lang)}
- {t('combined_outperform', lang)}

"""

        # 3. Minority class analysis
        report += f"""#### 3. {t('minority_improvement', lang)}

"""
        if exp07 is not None and exp08 is not None and exp09 is not None:
            can_07 = exp07.get('AP_Can', 0) or 0
            can_08 = exp08.get('AP_Can', 0) or 0
            can_09 = exp09.get('AP_Can', 0) or 0
            metal_07 = exp07.get('AP_Metal_Debris', 0) or 0
            metal_08 = exp08.get('AP_Metal_Debris', 0) or 0
            metal_09 = exp09.get('AP_Metal_Debris', 0) or 0

            report += f"""| {t('class', lang)} | {t('real_only', lang)} | {t('minority_focused', lang)} | {t('proportional', lang)} | Best |
|-------|-----------|-------------------|----------------------|------|
"""
            can_best = "Minority" if can_08 >= max(can_07, can_09) else ("Proportional" if can_09 >= can_07 else "Real-Only")
            metal_best = "Minority" if metal_08 >= max(metal_07, metal_09) else ("Proportional" if metal_09 >= metal_07 else "Real-Only")

            report += f"| Can | {can_07:.4f} | {can_08:.4f} ({_calc_improvement(can_07, can_08)}) | {can_09:.4f} ({_calc_improvement(can_07, can_09)}) | {can_best} |\n"
            report += f"| Metal_Debris | {metal_07:.4f} | {metal_08:.4f} ({_calc_improvement(metal_07, metal_08)}) | {metal_09:.4f} ({_calc_improvement(metal_07, metal_09)}) | {metal_best} |\n"

            report += f"""
**{t('conclusion', lang)}**: {t('targeted_synth', lang)}
{t('without_degrading', lang)} {metal_best.lower()} {t('best_balance', lang)}

"""

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    report += f"""---

## {t('recommendations', lang)}

### {t('production_deployment', lang)}

"""
    if has_data:
        report += f"""1. **{t('recommended_model', lang)}**: **{best_exp['exp_name']}** {t('use_for_production', lang)}
   - mAP@0.5:0.95: **{best_map:.4f}**
   - {t('best_balance_minority', lang)}

2. **{t('data_collection', lang)}**:
   - {t('focus_minority', lang)}
   - {t('even_10_pct', lang)}

3. **{t('synth_strategy', lang)}**:
   - {t('use_inverse_prop', lang)}
   - {t('generate_more_synth', lang)}
   - {t('avoid_synth_only', lang)}

4. **{t('training_config_rec', lang)}**:
   - {t('use_combined', lang)}
   - {t('apply_early_stop', lang)}
   - {t('monitor_minority', lang)}

### {t('future_work', lang)}

1. **{t('domain_adaptation', lang)}**
2. **{t('active_learning', lang)}**
3. **{t('data_augmentation', lang)}**
4. **{t('model_ensemble', lang)}**

"""

    # =========================================================================
    # APPENDIX - IMAGE SUMMARY
    # =========================================================================
    # Count how many images were found per experiment
    exp_with_images = []
    exp_without_images = []
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        if exp_name in copied_images and len(copied_images[exp_name]) > 0:
            exp_with_images.append(exp_name)
        else:
            exp_without_images.append(exp_name)

    report += f"""---

## Appendix: Image Availability Summary

| Status | Count | Experiments |
|--------|-------|-------------|
| ✅ With images | {len(exp_with_images)} | {', '.join(exp_with_images) if exp_with_images else 'None'} |
| ⚠️ No images found | {len(exp_without_images)} | {', '.join(exp_without_images) if exp_without_images else 'None'} |

"""
    if exp_without_images:
        report += f"""*Note: Experiments without images may not have been executed yet or their evaluation output is missing.*

"""

    # =========================================================================
    # APPENDIX - ALL EXPERIMENT IMAGES
    # =========================================================================
    report += f"""---

## {t('appendix_cm', lang)}

"""
    found_cm = False
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        if exp_name in copied_images and 'confusion_matrix_normalized' in copied_images[exp_name]:
            found_cm = True
            report += f"""### {exp_name}

![Confusion Matrix - {exp_name}]({copied_images[exp_name]['confusion_matrix_normalized']})

"""
        elif exp_name in copied_images and 'confusion_matrix' in copied_images[exp_name]:
            # Try unnormalized if normalized not available
            found_cm = True
            report += f"""### {exp_name}

![Confusion Matrix - {exp_name}]({copied_images[exp_name]['confusion_matrix']})

"""

    if not found_cm:
        report += "*No confusion matrix images found. Run experiments first to generate evaluation outputs.*\n\n"

    # =========================================================================
    # APPENDIX - SCORE DISTRIBUTIONS
    # =========================================================================
    report += f"""---

## {t('appendix_scores', lang)}

"""
    found_scores = False
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        if exp_name in copied_images and 'score_distributions' in copied_images[exp_name]:
            found_scores = True
            report += f"""### {exp_name}

![Score Distributions - {exp_name}]({copied_images[exp_name]['score_distributions']})

"""

    if not found_scores:
        report += "*No score distribution images found. Run experiments first to generate evaluation outputs.*\n\n"

    # =========================================================================
    # APPENDIX - PER-CLASS AP VISUALIZATIONS
    # =========================================================================
    report += f"""---

## Appendix: Per-Class AP Visualizations

"""
    found_ap = False
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        if exp_name in copied_images and 'per_class_ap' in copied_images[exp_name]:
            found_ap = True
            report += f"""### {exp_name}

![Per-Class AP - {exp_name}]({copied_images[exp_name]['per_class_ap']})

"""

    if not found_ap:
        report += "*No per-class AP images found. Run experiments first to generate evaluation outputs.*\n\n"

    # =========================================================================
    # APPENDIX - FP/FN BAR CHARTS
    # =========================================================================
    report += f"""---

## Appendix: False Positive/Negative Analysis Charts

"""
    found_fpfn = False
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        if exp_name in copied_images and 'fp_fn_bar_chart' in copied_images[exp_name]:
            found_fpfn = True
            report += f"""### {exp_name}

![FP/FN Chart - {exp_name}]({copied_images[exp_name]['fp_fn_bar_chart']})

"""

    if not found_fpfn:
        report += "*No FP/FN charts found. Run experiments first to generate evaluation outputs.*\n\n"

    # =========================================================================
    # APPENDIX - SIZE ANALYSIS
    # =========================================================================
    report += f"""---

## Appendix: AP/AR by Object Size

"""
    found_size = False
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        if exp_name in copied_images and 'ap_ar_by_size' in copied_images[exp_name]:
            found_size = True
            report += f"""### {exp_name}

![AP/AR by Size - {exp_name}]({copied_images[exp_name]['ap_ar_by_size']})

"""

    if not found_size:
        report += "*No size analysis images found. Run experiments first to generate evaluation outputs.*\n\n"

    # =========================================================================
    # APPENDIX - DATASET DISTRIBUTIONS
    # =========================================================================
    report += f"""---

## Appendix: Dataset Distributions

"""
    found_dist = False
    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        if exp_name in copied_images:
            imgs = copied_images[exp_name]
            if 'train_dataset_analysis' in imgs or 'val_dataset_analysis' in imgs:
                found_dist = True
                report += f"""### {exp_name}

"""
                if 'train_dataset_analysis' in imgs:
                    report += f"""#### Training Dataset Distribution

![Train Distribution - {exp_name}]({imgs['train_dataset_analysis']})

"""
                if 'val_dataset_analysis' in imgs:
                    report += f"""#### Validation Dataset Distribution

![Val Distribution - {exp_name}]({imgs['val_dataset_analysis']})

"""

    if not found_dist:
        report += "*No dataset distribution images found.*\n\n"

    # =========================================================================
    # FILES GENERATED
    # =========================================================================
    report += f"""---

## {t('files_reference', lang)}

### {t('main_results', lang)}
| {t('file', lang)} | {t('description', lang)} |
|------|-------------|
| `comparison_table.csv` | {t('complete_metrics_table', lang)} |
| `EXPERIMENT_REPORT.md` | {t('this_report', lang)} |

### {t('aggregated_viz', lang)}
| {t('file', lang)} | {t('description', lang)} |
|------|-------------|
| `mAP_vs_real_ratio.png` | {t('performance_progression', lang)} |
| `minority_class_analysis.png` | {t('minority_class_comparison', lang)} |
| `per_class_comparison.png` | {t('per_class_key_exp', lang)} |

### {t('per_exp_images', lang)}
{t('each_exp_contains', lang)}:
| {t('file', lang)} | {t('description', lang)} |
|------|-------------|
| `confusion_matrix_normalized.png` | {t('norm_conf_matrix', lang)} |
| `confusion_matrix.png` | {t('raw_conf_matrix', lang)} |
| `per_class_ap.png` | {t('per_class_ap_img', lang)} |
| `score_distributions.png` | {t('conf_score_dist', lang)} |
| `fp_fn_bar_chart.png` | {t('fp_fn_img', lang)} |
| `ap_ar_by_size.png` | {t('ap_ar_size_img', lang)} |
| `train_dataset_analysis.png` | {t('train_dist_img', lang)} |
| `val_dataset_analysis.png` | {t('val_dist_img', lang)} |

---

*{t('report_auto_gen', lang)}*
"""

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Saved: {output_path}")


# =============================================================================
# CROSS-VERSION COMPARISON
# =============================================================================

def collect_all_version_results(
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, Dict[str, Dict]]:
    """
    Collect results from all available synthetic dataset versions.

    Args:
        batch_mode: Batch sampling mode

    Returns:
        Dict mapping version name to experiment metrics dict
    """
    all_results = {}

    for version in get_available_synth_versions():
        versioned_paths = get_versioned_output_paths(version, batch_mode)
        results_dir = Path(versioned_paths['results'])

        # Check if version has any results
        state_file = results_dir / f"experiment_state_{version}.json"
        if state_file.exists():
            print(f"\n  Loading results for version: {version}")
            metrics = collect_all_metrics(version, batch_mode)

            # Only include if we have completed experiments
            completed = sum(1 for m in metrics.values() if m.get('status') == 'completed')
            if completed > 0:
                all_results[version] = metrics
                print(f"    {completed} completed experiments")
            else:
                print(f"    No completed experiments")
        else:
            print(f"  Skipping version {version}: no state file found")

    return all_results


def plot_cross_version_comparison(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Create bar chart comparing mAP across versions for each experiment,
    including Real-Only baselines where available.

    Args:
        all_results: Dict mapping version to experiment metrics
        output_path: Path to save the plot
    """
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available, skipping cross-version plot")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    # Get experiments that use synthetic data (1-9)
    exp_names = [exp['name'] for exp in EXPERIMENTS]
    versions = list(all_results.keys())

    if not versions:
        print("  No version results to compare")
        return

    # Mapping from synth+real experiment to its real-only counterpart
    real_only_pair = {
        'exp02_synth100_real10': 'exp11_real10',
        'exp03_synth100_real25': 'exp12_real25',
        'exp04_synth100_real50': 'exp13_real50',
        'exp05_synth100_real75': 'exp14_real75',
        'exp06_synth100_real100': 'exp07_real100',
    }

    # Prepare data
    data = {v: [] for v in versions}
    real_only_data = []
    first_version = versions[0]

    for exp_name in exp_names:
        for version in versions:
            metrics = all_results[version].get(exp_name, {})
            mAP = metrics.get('mAP', 0) if metrics.get('status') == 'completed' else 0
            data[version].append(mAP)

        # Get real-only baseline for this experiment
        real_pair = real_only_pair.get(exp_name)
        if real_pair:
            rm = all_results[first_version].get(real_pair, {})
            real_only_data.append(rm.get('mAP', 0) if rm.get('status') == 'completed' else 0)
        else:
            real_only_data.append(0)

    # Create plot - add 1 extra group for Real-Only
    n_groups = len(versions) + 1  # versions + real-only
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(exp_names))
    width = 0.8 / n_groups
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))

    # Plot Real-Only bars first
    offset_ro = (0 - n_groups/2 + 0.5) * width
    bars_ro = ax.bar(x + offset_ro, real_only_data, width,
                     label='Real-Only', color='#888888', edgecolor='#555555', linewidth=0.5)
    for bar, val in zip(bars_ro, real_only_data):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45,
                   color='#555555')

    # Plot version bars
    for i, (version, values) in enumerate(data.items()):
        offset = (i + 1 - n_groups/2 + 0.5) * width
        synth_config = get_synth_config(version)
        bars = ax.bar(x + offset, values, width, label=synth_config['name'],
                      color=colors[i], edgecolor=colors[i] * 0.8, linewidth=0.5)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('mAP@0.5:0.95')
    ax.set_title('Cross-Version Comparison: mAP by Synthetic Dataset Source\n(with Real-Only baselines)')
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace('exp', 'E') for e in exp_names], rotation=45, ha='right')
    ax.legend(title='Data Source', loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_version_ranking(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Create ranking heatmap showing which version performs best per experiment.

    Args:
        all_results: Dict mapping version to experiment metrics
        output_path: Path to save the plot
    """
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    exp_names = [exp['name'] for exp in EXPERIMENTS]
    versions = list(all_results.keys())

    if len(versions) < 2:
        print("  Need at least 2 versions to compare rankings")
        return

    # Build mAP matrix
    mAP_matrix = []
    for exp_name in exp_names:
        row = []
        for version in versions:
            metrics = all_results[version].get(exp_name, {})
            mAP = metrics.get('mAP', 0) if metrics.get('status') == 'completed' else 0
            row.append(mAP)
        mAP_matrix.append(row)

    mAP_matrix = np.array(mAP_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(mAP_matrix, cmap='YlGn', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('mAP@0.5:0.95', rotation=-90, va='bottom')

    # Set ticks
    version_labels = [get_synth_config(v)['name'] for v in versions]
    ax.set_xticks(np.arange(len(versions)))
    ax.set_yticks(np.arange(len(exp_names)))
    ax.set_xticklabels(version_labels)
    ax.set_yticklabels(exp_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    for i in range(len(exp_names)):
        for j in range(len(versions)):
            val = mAP_matrix[i, j]
            # Highlight best in row
            if val == mAP_matrix[i].max() and val > 0:
                text = ax.text(j, i, f'{val:.3f}*', ha='center', va='center',
                              color='darkgreen', fontweight='bold')
            elif val > 0:
                text = ax.text(j, i, f'{val:.3f}', ha='center', va='center')

    ax.set_title('Version Ranking by Experiment\n(* indicates best for that experiment)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


# =============================================================================
# CROSS-VERSION PLOT FUNCTIONS (NEW)
# =============================================================================

def plot_cross_version_map_curves(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Plot mAP vs real data ratio curves, one line per synthetic version."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    if not versions:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Synth+Real progression experiments (exp01-exp07)
    synth_real_exps = [e for e in EXPERIMENTS if e['id'] <= 7]
    ratios = [e['real_ratio'] for e in synth_real_exps]
    exp_names = [e['name'] for e in synth_real_exps]

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))
    markers = ['o', 's', '^', 'D', 'v']

    for i, version in enumerate(versions):
        maps = []
        for exp_name in exp_names:
            m = all_results[version].get(exp_name, {})
            val = m.get('mAP', 0) if m.get('status') == 'completed' else 0
            maps.append(val)

        synth_config = get_synth_config(version)
        marker = markers[i % len(markers)]
        ax.plot([r * 100 for r in ratios], maps,
                f'-{marker}', linewidth=2.5, markersize=9,
                label=f"{synth_config['name']} ({synth_config['num_classes']} cls)",
                color=colors[i])
        for r, m_val in zip(ratios, maps):
            if m_val > 0:
                ax.annotate(f'{m_val:.3f}', (r * 100, m_val),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=7, color=colors[i])

    # Real-only progression curve (exp11→exp12→exp13→exp14→exp07)
    first_version = versions[0]
    real_only_exps = [
        ('exp11_real10', 0.1), ('exp12_real25', 0.25),
        ('exp13_real50', 0.5), ('exp14_real75', 0.75),
        ('exp07_real100', 1.0),
    ]
    real_ratios_line = []
    real_maps_line = []
    for exp_name, ratio in real_only_exps:
        m = all_results[first_version].get(exp_name, {})
        val = m.get('mAP', 0) if m.get('status') == 'completed' else 0
        if val > 0:
            real_ratios_line.append(ratio * 100)
            real_maps_line.append(val)

    if real_maps_line:
        ax.plot(real_ratios_line, real_maps_line, '--D', linewidth=2.5, markersize=9,
                label='Real-Only (no synthetic)', color='#555555')
        for r, m_val in zip(real_ratios_line, real_maps_line):
            ax.annotate(f'{m_val:.3f}', (r, m_val),
                       textcoords="offset points", xytext=(0, -15),
                       ha='center', fontsize=7, color='#555555')

    # Fill area between best synth+real line and real-only to show benefit
    if real_maps_line and len(versions) > 0:
        # Use the best-performing version for the fill
        best_v = max(versions, key=lambda v: sum(
            all_results[v].get(e['name'], {}).get('mAP', 0)
            for e in synth_real_exps if all_results[v].get(e['name'], {}).get('status') == 'completed')
        )
        best_maps = []
        best_ratios_fill = []
        for exp_name, ratio in real_only_exps:
            synth_exp = {0.1: 'exp02_synth100_real10', 0.25: 'exp03_synth100_real25',
                         0.5: 'exp04_synth100_real50', 0.75: 'exp05_synth100_real75',
                         1.0: 'exp06_synth100_real100'}.get(ratio)
            if synth_exp:
                m = all_results[best_v].get(synth_exp, {})
                val = m.get('mAP', 0) if m.get('status') == 'completed' else 0
                if val > 0:
                    best_maps.append(val)
                    best_ratios_fill.append(ratio * 100)

        if len(best_maps) == len(real_maps_line):
            ax.fill_between(real_ratios_line, real_maps_line, best_maps,
                           alpha=0.1, color='#3498db', label='Synthetic benefit zone')

    ax.set_xlabel('Percentage of Real Data (%)', fontsize=12)
    ax.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax.set_title('Cross-Version: mAP vs Real Data Ratio\n(Synth+Real versions vs Real-Only baseline)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-5, 105)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_per_class_heatmap(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Heatmap of per-class AP across versions for key experiments."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    key_exps = ['exp01_synth100', 'exp07_real100', 'exp06_synth100_real100', 'exp08_real_synth_minor', 'exp09_real_synth_prop']
    key_exp_labels = ['Synth 100%', 'Real 100%', 'Synth+Real 100%', 'Minority Strategy', 'Proportional Strategy']

    # Build column labels: version x experiment (Real-Only shown once since it's shared)
    col_labels = []
    col_exp_version = []  # track which (exp, version) each column maps to
    for exp_name, exp_label in zip(key_exps, key_exp_labels):
        if exp_name == 'exp07_real100':
            # Real-only: show once (shared baseline)
            col_labels.append(f"{exp_label}\n(Baseline)")
            col_exp_version.append((exp_name, versions[0]))
        else:
            for version in versions:
                synth_config = get_synth_config(version)
                col_labels.append(f"{exp_label}\n{synth_config['name']}")
                col_exp_version.append((exp_name, version))

    # Build data matrix: rows=classes, cols as defined above
    data = []
    for cls_name in CLASS_NAMES:
        row = []
        for exp_name, version in col_exp_version:
            m = all_results[version].get(exp_name, {})
            ap = m.get(f'AP_{cls_name}', 0) if m.get('status') == 'completed' else 0
            row.append(ap if ap else 0)
        data.append(row)

    data = np.array(data)
    if data.max() == 0:
        print(f"  Skipped {output_path.name}: No per-class AP data")
        return

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 1.2), 8))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('AP@0.5:0.95', rotation=-90, va='bottom')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotate cells
    for i in range(len(CLASS_NAMES)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if val > 0:
                color = 'white' if val > data.max() * 0.65 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6, color=color)

    ax.set_title('Per-Class AP Across Versions and Key Experiments', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_radar(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Radar/spider chart comparing versions on multiple metrics."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    if not versions:
        return

    # Use the best mixed experiment (exp06 or exp09) for each version
    best_exp = 'exp09_real_synth_prop'
    metric_keys = ['mAP', 'mAP_50', 'mAP_75', 'AP_small', 'AP_medium', 'AP_large']
    metric_labels = ['mAP', 'mAP@50', 'mAP@75', 'AP Small', 'AP Medium', 'AP Large']

    # Collect data
    version_data = {}
    has_data = False
    for version in versions:
        m = all_results[version].get(best_exp, {})
        if m.get('status') != 'completed':
            m = all_results[version].get('exp06_synth100_real100', {})
        values = []
        for key in metric_keys:
            val = m.get(key, 0) or 0
            values.append(val)
        version_data[version] = values
        if any(v > 0 for v in values):
            has_data = True

    if not has_data:
        print(f"  Skipped {output_path.name}: No multi-metric data")
        return

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))

    for i, version in enumerate(versions):
        values = version_data[version] + version_data[version][:1]
        synth_config = get_synth_config(version)
        ax.plot(angles, values, '-o', linewidth=2, markersize=6,
                label=synth_config['name'], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_title(f'Multi-Metric Comparison ({best_exp})', fontsize=13, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_minority_analysis(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Bar chart comparing minority class AP across versions for key experiments."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    key_exps = ['exp01_synth100', 'exp06_synth100_real100', 'exp07_real100',
                'exp08_real_synth_minor', 'exp09_real_synth_prop']
    key_labels = ['Synth 100%', 'Synth+Real', 'Real 100%', 'Minority', 'Proportional']
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for cls_idx, (cls_name, ax) in enumerate(zip(['Can', 'Metal_Debris'], axes)):
        x = np.arange(len(key_exps))
        width = 0.8 / max(len(versions), 1)

        for i, version in enumerate(versions):
            ap_vals = []
            for exp_name in key_exps:
                m = all_results[version].get(exp_name, {})
                ap = m.get(f'AP_{cls_name}', 0) if m.get('status') == 'completed' else 0
                ap_vals.append(ap if ap else 0)

            synth_config = get_synth_config(version)
            offset = (i - len(versions)/2 + 0.5) * width
            bars = ax.bar(x + offset, ap_vals, width, label=synth_config['name'], color=colors[i])
            for bar, val in zip(bars, ap_vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_title(f'{cls_name} (Minority Class)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(key_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('AP@0.5:0.95')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Minority Class AP Across Synthetic Versions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_training_efficiency(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Bar chart comparing training duration across versions."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    exp_names = [e['name'] for e in EXPERIMENTS if e['id'] <= 7]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(exp_names))
    width = 0.8 / max(len(versions), 1)
    has_data = False

    for i, version in enumerate(versions):
        durations = []
        for exp_name in exp_names:
            m = all_results[version].get(exp_name, {})
            training = m.get('training', {})
            dur_s = 0
            if isinstance(training, dict):
                dur_s = training.get('duration_seconds', 0) or 0
            durations.append(dur_s / 3600)  # Convert to hours
            if dur_s > 0:
                has_data = True

        synth_config = get_synth_config(version)
        offset = (i - len(versions)/2 + 0.5) * width
        ax.bar(x + offset, durations, width, label=synth_config['name'], color=colors[i])

    if not has_data:
        plt.close()
        print(f"  Skipped {output_path.name}: No training duration data")
        return

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Training Duration (hours)')
    ax.set_title('Training Duration Comparison Across Versions')
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace('exp0', 'E') for e in exp_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_domain_gap(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Visualize domain gap (synth-only vs real-only) and recovery ratio per version."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Domain gap bars
    gaps = []
    synth_maps = []
    real_maps = []
    v_labels = []
    for version in versions:
        synth_m = all_results[version].get('exp01_synth100', {})
        real_m = all_results[version].get('exp07_real100', {})
        s_map = synth_m.get('mAP', 0) if synth_m.get('status') == 'completed' else 0
        r_map = real_m.get('mAP', 0) if real_m.get('status') == 'completed' else 0
        synth_maps.append(s_map)
        real_maps.append(r_map)
        gaps.append(r_map - s_map)
        v_labels.append(get_synth_config(version)['name'])

    x = np.arange(len(versions))
    width = 0.3
    ax1.bar(x - width/2, synth_maps, width, label='Synth-Only (exp01)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, real_maps, width, label='Real-Only (exp07)', color='#27ae60', alpha=0.8)
    for i, (s, r) in enumerate(zip(synth_maps, real_maps)):
        if s > 0 and r > 0:
            ax1.annotate(f'Gap: {r-s:.3f}', (i, max(s, r) + 0.01),
                        ha='center', fontsize=9, fontweight='bold', color='#e67e22')
    ax1.set_xticks(x)
    ax1.set_xticklabels(v_labels, rotation=30, ha='right')
    ax1.set_ylabel('mAP@0.5:0.95')
    ax1.set_title('Domain Gap: Synth-Only vs Real-Only')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right: Recovery ratio (how much of real performance is recovered by adding synth)
    recovery_data = []
    for version in versions:
        combined_m = all_results[version].get('exp06_synth100_real100', {})
        real_m = all_results[version].get('exp07_real100', {})
        c_map = combined_m.get('mAP', 0) if combined_m.get('status') == 'completed' else 0
        r_map = real_m.get('mAP', 0) if real_m.get('status') == 'completed' else 0
        ratio = (c_map / r_map * 100) if r_map > 0 else 0
        recovery_data.append(ratio)

    bars = ax2.bar(x, recovery_data, 0.5, color=colors[:len(versions)])
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='100% Recovery')
    for bar, val in zip(bars, recovery_data):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(v_labels, rotation=30, ha='right')
    ax2.set_ylabel('Recovery Ratio (%)')
    ax2.set_title('Performance Recovery: Combined(Synth+Real) / Real-Only')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Domain Gap Analysis Across Synthetic Versions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_size_analysis(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Bar chart of AP by object size across versions for the best experiment."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    best_exp = 'exp09_real_synth_prop'
    size_metrics = ['AP_small', 'AP_medium', 'AP_large']
    size_labels = ['Small (<32²)', 'Medium (32²-96²)', 'Large (>96²)']
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(size_metrics))
    width = 0.8 / max(len(versions), 1)
    has_data = False

    for i, version in enumerate(versions):
        m = all_results[version].get(best_exp, {})
        if m.get('status') != 'completed':
            m = all_results[version].get('exp06_synth100_real100', {})
        values = []
        for key in size_metrics:
            val = m.get(key, 0) or 0
            values.append(val)
            if val > 0:
                has_data = True

        synth_config = get_synth_config(version)
        offset = (i - len(versions)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=synth_config['name'], color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    if not has_data:
        plt.close()
        print(f"  Skipped {output_path.name}: No size-based AP data")
        return

    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.set_ylabel('AP@0.5:0.95')
    ax.set_title(f'AP by Object Size Across Versions ({best_exp})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_synth_benefit(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Line chart showing the benefit of adding synthetic data at each real ratio, per version."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    # Paired comparisons: synth+real vs real-only at matching ratios
    pairs = [
        ('exp02_synth100_real10', 'exp11_real10', '10%'),
        ('exp03_synth100_real25', 'exp12_real25', '25%'),
        ('exp04_synth100_real50', 'exp13_real50', '50%'),
        ('exp05_synth100_real75', 'exp14_real75', '75%'),
        ('exp06_synth100_real100', 'exp07_real100', '100%'),
    ]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))
    markers = ['o', 's', '^', 'D', 'v']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Delta mAP (synth+real - real-only)
    for i, version in enumerate(versions):
        deltas = []
        ratios = []
        for synth_exp, real_exp, label in pairs:
            sm = all_results[version].get(synth_exp, {})
            rm = all_results[version].get(real_exp, {})
            s_map = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r_map = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            delta = s_map - r_map if (s_map > 0 and r_map > 0) else 0
            deltas.append(delta)
            ratios.append(label)

        synth_config = get_synth_config(version)
        marker = markers[i % len(markers)]
        ax1.plot(range(len(ratios)), deltas, f'-{marker}', linewidth=2.5, markersize=9,
                label=synth_config['name'], color=colors[i])
        for j, d in enumerate(deltas):
            if d != 0:
                ax1.annotate(f'{d:+.3f}', (j, d), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8, color=colors[i])

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xticks(range(len(pairs)))
    ax1.set_xticklabels([p[2] for p in pairs])
    ax1.set_xlabel('Real Data %')
    ax1.set_ylabel('Δ mAP (Synth+Real minus Real-Only)')
    ax1.set_title('Synthetic Data Benefit by Version')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Relative improvement (%)
    for i, version in enumerate(versions):
        improvements = []
        for synth_exp, real_exp, label in pairs:
            sm = all_results[version].get(synth_exp, {})
            rm = all_results[version].get(real_exp, {})
            s_map = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r_map = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            pct = ((s_map - r_map) / r_map * 100) if r_map > 0 and s_map > 0 else 0
            improvements.append(pct)

        synth_config = get_synth_config(version)
        marker = markers[i % len(markers)]
        ax2.plot(range(len(pairs)), improvements, f'-{marker}', linewidth=2.5, markersize=9,
                label=synth_config['name'], color=colors[i])

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xticks(range(len(pairs)))
    ax2.set_xticklabels([p[2] for p in pairs])
    ax2.set_xlabel('Real Data %')
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.set_title('Relative Benefit of Synthetic Data (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Synthetic Data Benefit Analysis Across Versions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_version_paired_comparison(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Grouped bar chart: Real-Only vs each synthetic version at each real data ratio."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    versions = list(all_results.keys())
    first_v = versions[0]

    pairs = [
        ('exp11_real10', 'exp02_synth100_real10', '10%'),
        ('exp12_real25', 'exp03_synth100_real25', '25%'),
        ('exp13_real50', 'exp04_synth100_real50', '50%'),
        ('exp14_real75', 'exp05_synth100_real75', '75%'),
        ('exp07_real100', 'exp06_synth100_real100', '100%'),
    ]

    # Number of groups: len(pairs)
    # Bars per group: 1 (real-only) + len(versions) (synth+real per version)
    n_bars = 1 + len(versions)
    group_width = 0.8
    bar_width = group_width / n_bars

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(pairs))

    # Real-only bars (shared baseline)
    real_vals = []
    for real_exp, _, _ in pairs:
        m = all_results[first_v].get(real_exp, {})
        val = m.get('mAP', 0) if m.get('status') == 'completed' else 0
        real_vals.append(val)

    offset = -group_width / 2 + bar_width / 2
    bars_real = ax.bar(x + offset, real_vals, bar_width, label='Real-Only',
                       color='#555555', edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars_real, real_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    # Synth+Real bars per version
    version_colors = plt.cm.Set2(np.linspace(0, 1, max(len(versions), 3)))
    for i, version in enumerate(versions):
        synth_vals = []
        for _, synth_exp, _ in pairs:
            m = all_results[version].get(synth_exp, {})
            val = m.get('mAP', 0) if m.get('status') == 'completed' else 0
            synth_vals.append(val)

        sc = get_synth_config(version)
        bar_offset = offset + (i + 1) * bar_width
        bars_v = ax.bar(x + bar_offset, synth_vals, bar_width,
                        label=f'Synth+Real ({sc["name"]})',
                        color=version_colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars_v, synth_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    # Add delta annotations
    for j, (_, synth_exp, label) in enumerate(pairs):
        real_val = real_vals[j]
        if real_val > 0:
            for i, version in enumerate(versions):
                m = all_results[version].get(synth_exp, {})
                synth_val = m.get('mAP', 0) if m.get('status') == 'completed' else 0
                if synth_val > 0:
                    delta = synth_val - real_val
                    bar_offset = offset + (i + 1) * bar_width
                    color = '#27ae60' if delta > 0 else '#e74c3c'
                    ax.annotate(f'{delta:+.3f}', (j + bar_offset, synth_val + 0.025),
                               ha='center', fontsize=6, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([p[2] + ' Real Data' for p in pairs], fontsize=11)
    ax.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax.set_title('Paired Comparison: Real-Only vs Synth+Real at Each Data Level\n'
                 '(Green/red numbers show Δ mAP from adding synthetic data)', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# CROSS-VERSION INSIGHTS COMPUTATION
# =============================================================================

def _compute_cross_version_insights(
    all_results: Dict[str, Dict[str, Dict]]
) -> Dict[str, Any]:
    """Compute dynamic insights from cross-version results."""
    versions = list(all_results.keys())
    exp_names = [e['name'] for e in EXPERIMENTS]
    insights = {}

    # --- Win counts ---
    win_counts = {v: 0 for v in versions}
    win_margins = []
    for exp_name in exp_names:
        maps = {}
        for version in versions:
            m = all_results[version].get(exp_name, {})
            maps[version] = m.get('mAP', 0) if m.get('status') == 'completed' else 0
        if any(v > 0 for v in maps.values()):
            best_v = max(maps, key=maps.get)
            win_counts[best_v] += 1
            sorted_vals = sorted(maps.values(), reverse=True)
            if len(sorted_vals) > 1 and sorted_vals[0] > 0:
                win_margins.append(sorted_vals[0] - sorted_vals[1])

    insights['win_counts'] = win_counts
    insights['margin_of_victory'] = sum(win_margins) / len(win_margins) if win_margins else 0

    # Dominant version
    total_exps = sum(1 for e in exp_names
                     if any(all_results[v].get(e, {}).get('status') == 'completed' for v in versions))
    best_version = max(win_counts, key=win_counts.get) if win_counts else None
    insights['dominant_version'] = best_version
    insights['is_dominant'] = (win_counts.get(best_version, 0) / max(total_exps, 1)) > 0.7 if best_version else False

    # --- Average mAP per version ---
    avg_maps = {}
    for version in versions:
        maps = []
        for exp_name in exp_names:
            m = all_results[version].get(exp_name, {})
            if m.get('status') == 'completed' and m.get('mAP', 0) > 0:
                maps.append(m['mAP'])
        avg_maps[version] = sum(maps) / len(maps) if maps else 0
    insights['avg_maps'] = avg_maps

    # --- Domain gap per version ---
    domain_gaps = {}
    for version in versions:
        synth_m = all_results[version].get('exp01_synth100', {})
        real_m = all_results[version].get('exp07_real100', {})
        combined_m = all_results[version].get('exp06_synth100_real100', {})
        s_map = synth_m.get('mAP', 0) if synth_m.get('status') == 'completed' else 0
        r_map = real_m.get('mAP', 0) if real_m.get('status') == 'completed' else 0
        c_map = combined_m.get('mAP', 0) if combined_m.get('status') == 'completed' else 0
        gap_abs = r_map - s_map
        gap_pct = (gap_abs / r_map * 100) if r_map > 0 else 0
        recovery = (c_map / r_map * 100) if r_map > 0 else 0
        domain_gaps[version] = {
            'synth_map': s_map, 'real_map': r_map, 'combined_map': c_map,
            'gap_abs': gap_abs, 'gap_pct': gap_pct, 'recovery_ratio': recovery
        }
    insights['domain_gaps'] = domain_gaps

    # --- Best per class ---
    best_per_class = {}
    best_exp_for_class = 'exp09_real_synth_prop'
    for cls_name in CLASS_NAMES:
        best_v = None
        best_ap = 0
        for version in versions:
            m = all_results[version].get(best_exp_for_class, {})
            if m.get('status') != 'completed':
                m = all_results[version].get('exp06_synth100_real100', {})
            ap = m.get(f'AP_{cls_name}', 0) or 0
            if ap > best_ap:
                best_ap = ap
                best_v = version
        best_per_class[cls_name] = {'version': best_v, 'ap': best_ap}
    insights['best_per_class'] = best_per_class

    # --- Minority class best ---
    minority_best = {}
    for cls_name in MINORITY_CLASSES:
        best_v = None
        best_ap = 0
        for version in versions:
            for exp_name in exp_names:
                m = all_results[version].get(exp_name, {})
                ap = m.get(f'AP_{cls_name}', 0) or 0
                if ap > best_ap:
                    best_ap = ap
                    best_v = version
        minority_best[cls_name] = best_v
    insights['minority_best'] = minority_best

    # --- Size analysis ---
    size_analysis = {}
    for version in versions:
        m = all_results[version].get(best_exp_for_class, {})
        if m.get('status') != 'completed':
            m = all_results[version].get('exp06_synth100_real100', {})
        size_analysis[version] = {
            'AP_small': m.get('AP_small', 0) or 0,
            'AP_medium': m.get('AP_medium', 0) or 0,
            'AP_large': m.get('AP_large', 0) or 0,
        }
    insights['size_analysis'] = size_analysis

    # --- Training efficiency ---
    training_efficiency = {}
    for version in versions:
        durations = []
        for exp_name in exp_names:
            m = all_results[version].get(exp_name, {})
            training = m.get('training', {})
            if isinstance(training, dict):
                dur = training.get('duration_seconds', 0) or 0
                if dur > 0:
                    durations.append(dur)
        avg_hours = (sum(durations) / len(durations) / 3600) if durations else 0
        training_efficiency[version] = avg_hours
    insights['training_efficiency'] = training_efficiency

    # --- Synth benefit by version ---
    synth_benefit = {}
    pairs = [
        ('exp02_synth100_real10', 'exp11_real10', 0.1),
        ('exp03_synth100_real25', 'exp12_real25', 0.25),
        ('exp04_synth100_real50', 'exp13_real50', 0.5),
        ('exp05_synth100_real75', 'exp14_real75', 0.75),
        ('exp06_synth100_real100', 'exp07_real100', 1.0),
    ]
    for version in versions:
        benefit = {}
        for synth_exp, real_exp, ratio in pairs:
            sm = all_results[version].get(synth_exp, {})
            rm = all_results[version].get(real_exp, {})
            s_map = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r_map = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            benefit[ratio] = s_map - r_map if (s_map > 0 and r_map > 0) else 0
        synth_benefit[version] = benefit
    insights['synth_benefit'] = synth_benefit

    # --- Surprising findings ---
    surprising = []
    trade_offs = []

    # Check if fewer-class version ever beats more-class version
    for exp_name in exp_names:
        maps = {}
        for version in versions:
            m = all_results[version].get(exp_name, {})
            maps[version] = m.get('mAP', 0) if m.get('status') == 'completed' else 0
        if len(maps) >= 2:
            sorted_versions = sorted(maps.items(), key=lambda x: -x[1])
            winner = sorted_versions[0]
            if winner[1] > 0:
                winner_cls = get_synth_config(winner[0])['num_classes']
                for loser_v, loser_map in sorted_versions[1:]:
                    loser_cls = get_synth_config(loser_v)['num_classes']
                    if loser_cls > winner_cls and loser_map > 0:
                        surprising.append(
                            f"{get_synth_config(winner[0])['name']} ({winner_cls} classes) "
                            f"beats {get_synth_config(loser_v)['name']} ({loser_cls} classes) "
                            f"in {exp_name} ({winner[1]:.3f} vs {loser_map:.3f})"
                        )

    # Check for trade-offs: one version wins mAP but another wins per-class
    if len(versions) >= 2:
        overall_winner = insights['dominant_version']
        other_versions = [v for v in versions if v != overall_winner]
        for other_v in other_versions:
            classes_won_by_other = []
            for cls_name, info in best_per_class.items():
                if info['version'] == other_v and info['ap'] > 0:
                    classes_won_by_other.append(cls_name)
            if classes_won_by_other:
                trade_offs.append(
                    f"{get_synth_config(other_v)['name']} is best for: {', '.join(classes_won_by_other)}"
                )

    insights['surprising_findings'] = surprising
    insights['trade_offs'] = trade_offs

    return insights


# =============================================================================
# FULL CROSS-VERSION REPORT (i18n)
# =============================================================================

def generate_cross_version_report_full(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    lang: str = 'en'
) -> None:
    """Generate comprehensive cross-version comparison report with i18n support."""
    versions = list(all_results.keys())
    if not versions:
        print(f"  No data for cross-version report")
        return

    insights = _compute_cross_version_insights(all_results)
    exp_names = [e['name'] for e in EXPERIMENTS]
    all_exp_names = [e['name'] for e in ALL_EXPERIMENTS]

    # =========================================================================
    # 1. HEADER & TOC
    # =========================================================================
    report = f"""# {t('cv_title', lang)}

### {t('cv_subtitle', lang)}

**{t('cv_generated', lang)}:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## {t('cv_toc', lang)}

1. [{t('cv_executive_summary', lang)}](#1-{t('cv_executive_summary', lang).lower().replace(' ', '-')})
2. [{t('cv_methodology', lang)}](#2-{t('cv_methodology', lang).lower().replace(' ', '-')})
3. [{t('cv_map_comparison', lang)}](#3-{t('cv_map_comparison', lang).lower().replace(' ', '-')})
4. [{t('cv_map_curves', lang)}](#4-{t('cv_map_curves', lang).lower().replace(' ', '-')})
5. [{t('cv_domain_gap_analysis', lang)}](#5-{t('cv_domain_gap_analysis', lang).lower().replace(' ', '-')})
6. [{t('cv_per_class_analysis', lang)}](#6-{t('cv_per_class_analysis', lang).lower().replace(' ', '-')})
7. [{t('cv_minority_analysis', lang)}](#7-{t('cv_minority_analysis', lang).lower().replace(' ', '-')})
8. [{t('cv_radar_chart', lang)}](#8-{t('cv_radar_chart', lang).lower().replace(' ', '-')})
9. [{t('cv_size_analysis', lang)}](#9-{t('cv_size_analysis', lang).lower().replace(' ', '-')})
10. [{t('cv_training_efficiency', lang)}](#10-{t('cv_training_efficiency', lang).lower().replace(' ', '-')})
11. [{t('cv_synth_benefit', lang)}](#11-{t('cv_synth_benefit', lang).lower().replace(' ', '-')})
12. [{t('cv_paired_comparison', lang)}](#12-{t('cv_paired_comparison', lang).lower().replace(' ', '-')})
13. [{t('cv_ranking', lang)}](#13-{t('cv_ranking', lang).lower().replace(' ', '-')})
14. [{t('cv_conclusions', lang)}](#14-{t('cv_conclusions', lang).lower().replace(' ', '-')})
15. [{t('cv_recommendations', lang)}](#15-{t('cv_recommendations', lang).lower().replace(' ', '-')})
16. [{t('cv_files_reference', lang)}](#16-{t('cv_files_reference', lang).lower().replace(' ', '-')})

---

"""

    # =========================================================================
    # 2. EXECUTIVE SUMMARY
    # =========================================================================
    report += f"## 1. {t('cv_executive_summary', lang)}\n\n"

    # Key results table
    report += f"### {t('key_results', lang)}\n\n"
    report += f"| {t('metric', lang)} | {t('value', lang)} |\n|---|---|\n"

    dominant = insights['dominant_version']
    dominant_name = get_synth_config(dominant)['name'] if dominant else 'N/A'
    report += f"| {t('cv_overall_winner', lang)} | **{dominant_name}** |\n"
    report += f"| {t('cv_wins_by_version', lang)} | "
    report += ", ".join(f"{get_synth_config(v)['name']}: {c}" for v, c in insights['win_counts'].items())
    report += " |\n"
    report += f"| {t('cv_improvement_margin', lang)} | {insights['margin_of_victory']:.4f} mAP |\n"

    for version in versions:
        synth_config = get_synth_config(version)
        report += f"| {t('cv_avg_map', lang)} ({synth_config['name']}) | {insights['avg_maps'].get(version, 0):.4f} |\n"

    report += f"| {t('num_experiments', lang)} | {len(exp_names)} + {len(REAL_ONLY_EXPERIMENTS)} baselines |\n"
    report += "\n"

    # Critical findings (dynamic)
    report += f"### {t('critical_findings', lang)}\n\n"

    if insights['is_dominant']:
        report += f"1. **{get_synth_config(dominant)['name']}** {t('cv_conclusion_dominant', lang)} "
        report += f"({insights['win_counts'][dominant]}/{len(exp_names)} {t('cv_wins_by_version', lang).lower()}).\n"
    else:
        report += f"1. {t('cv_competitive_versions', lang)}: "
        report += ", ".join(f"{get_synth_config(v)['name']} ({c} wins)" for v, c in insights['win_counts'].items())
        report += ".\n"

    # Domain gap insight
    gaps = insights['domain_gaps']
    if gaps:
        min_gap_v = min(gaps, key=lambda v: gaps[v]['gap_pct'] if gaps[v]['gap_pct'] > 0 else float('inf'))
        report += f"2. {t('cv_conclusion_domain_gap', lang)}: **{get_synth_config(min_gap_v)['name']}** "
        report += f"({gaps[min_gap_v]['gap_pct']:.1f}% gap vs "
        other_gaps = [f"{get_synth_config(v)['name']}: {gaps[v]['gap_pct']:.1f}%" for v in versions if v != min_gap_v and gaps[v]['gap_pct'] > 0]
        report += ", ".join(other_gaps) + ").\n"

    # Minority class insight
    minority_best = insights.get('minority_best', {})
    if minority_best:
        report += f"3. {t('cv_conclusion_minority', lang)}: "
        for cls_name, best_v in minority_best.items():
            if best_v:
                report += f"**{cls_name}** -> {get_synth_config(best_v)['name']}, "
        report = report.rstrip(', ') + ".\n"

    # Surprising findings
    for finding in insights.get('surprising_findings', [])[:2]:
        report += f"4. {t('cv_surprising_finding', lang)}: {finding}\n"

    report += "\n---\n\n"

    # =========================================================================
    # 3. METHODOLOGY
    # =========================================================================
    report += f"## 2. {t('cv_methodology', lang)}\n\n"
    report += f"### {t('cv_versions_compared', lang)}\n\n"
    report += f"| {t('cv_version', lang)} | {t('cv_name', lang)} | {t('cv_num_classes', lang)} | {t('cv_missing_classes', lang)} | {t('description', lang)} |\n"
    report += "|---|---|---|---|---|\n"

    for version in versions:
        sc = get_synth_config(version)
        missing = sc.get('missing_real_classes', [])
        missing_names = [CLASSES.get(c, str(c)) for c in missing] if missing else [t('cv_none', lang)]
        report += f"| {version} | {sc['name']} | {sc['num_classes']} | {', '.join(missing_names)} | {sc['description']} |\n"

    report += f"\n### {t('cv_exp_design', lang)}\n\n"
    report += f"{t('cv_exp_design_desc', lang)}\n\n"
    report += f"| ID | {t('experiment', lang)} | {t('description', lang)} |\n|---|---|---|\n"
    for exp in EXPERIMENTS:
        report += f"| {exp['id']} | {exp['name']} | {exp['description']} |\n"

    report += "\n---\n\n"

    # =========================================================================
    # 4. mAP COMPARISON TABLE (with Real-Only baseline)
    # =========================================================================
    report += f"## 3. {t('cv_map_comparison', lang)}\n\n"
    report += f"{t('cv_real_only_comparison_note', lang)}\n\n"

    # Mapping from synth+real experiment to its real-only counterpart
    real_only_pair = {
        'exp02_synth100_real10': 'exp11_real10',
        'exp03_synth100_real25': 'exp12_real25',
        'exp04_synth100_real50': 'exp13_real50',
        'exp05_synth100_real75': 'exp14_real75',
        'exp06_synth100_real100': 'exp07_real100',
        'exp08_real_synth_minor': 'exp07_real100',
        'exp09_real_synth_prop': 'exp07_real100',
    }

    # Build header: Experiment | Real-Only | Version1 | Δ vs Real | Version2 | Δ vs Real | ...
    report += f"| {t('experiment', lang)} | {t('cv_real_only_baseline', lang)} | "
    for version in versions:
        sc = get_synth_config(version)
        report += f"{sc['name']} | {t('cv_delta_vs_real', lang)} | "
    report += "\n|---|---|"
    for _ in versions:
        report += "---|---|"
    report += "\n"

    # Only show experiments that use synthetic data (exp01-exp09), not the real-only baselines
    for exp_name in exp_names:
        report += f"| {exp_name} | "

        # Get real-only baseline for this experiment
        real_pair = real_only_pair.get(exp_name)
        first_version = versions[0]
        if real_pair:
            real_m = all_results[first_version].get(real_pair, {})
            real_val = real_m.get('mAP', 0) if real_m.get('status') == 'completed' else 0
            report += f"{real_val:.4f} | " if real_val > 0 else "— | "
        else:
            real_val = 0
            report += "— | "

        for version in versions:
            m = all_results[version].get(exp_name, {})
            val = m.get('mAP', 0) if m.get('status') == 'completed' else 0

            if val > 0:
                report += f"{val:.4f} | "
            else:
                report += "N/A | "

            # Delta vs real-only baseline
            if val > 0 and real_val > 0:
                delta = val - real_val
                delta_str = f"**{delta:+.4f}**" if delta > 0 else f"{delta:+.4f}"
                report += f"{delta_str} | "
            else:
                report += "— | "

        report += "\n"

    report += "\n![Cross-Version mAP Comparison](cross_version_comparison.png)\n\n"
    report += "---\n\n"

    # =========================================================================
    # 5. mAP CURVES
    # =========================================================================
    report += f"## 4. {t('cv_map_curves', lang)}\n\n"
    report += f"{t('cv_real_ratio_progression', lang)}.\n\n"
    report += f"{t('cv_real_only_curve_note', lang)}\n\n"
    report += "![Cross-Version mAP Curves](cross_version_map_curves.png)\n\n"

    # Dynamic observations
    report += f"### {t('key_observations', lang)}\n\n"

    # Synth+Real progression per version
    for version in versions:
        sc = get_synth_config(version)
        synth_m = all_results[version].get('exp01_synth100', {})
        combined_m = all_results[version].get('exp06_synth100_real100', {})
        s_map = synth_m.get('mAP', 0) if synth_m.get('status') == 'completed' else 0
        c_map = combined_m.get('mAP', 0) if combined_m.get('status') == 'completed' else 0
        if s_map > 0 and c_map > 0:
            improvement = ((c_map - s_map) / s_map * 100) if s_map > 0 else 0
            report += f"- **{sc['name']}**: {t('cv_synth_only_map', lang)} = {s_map:.3f} -> "
            report += f"Synth+Real 100% = {c_map:.3f} ({improvement:+.1f}%)\n"

    # Real-Only progression
    first_v = versions[0]
    real_only_prog = [
        ('exp11_real10', '10%'), ('exp12_real25', '25%'),
        ('exp13_real50', '50%'), ('exp14_real75', '75%'),
        ('exp07_real100', '100%'),
    ]
    real_vals = []
    for rexp, _ in real_only_prog:
        rm = all_results[first_v].get(rexp, {})
        rv = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
        real_vals.append(rv)
    if real_vals[0] > 0 and real_vals[-1] > 0:
        real_impr = ((real_vals[-1] - real_vals[0]) / real_vals[0] * 100) if real_vals[0] > 0 else 0
        report += f"- **Real-Only**: {real_vals[0]:.3f} (10%) -> {real_vals[-1]:.3f} (100%) ({real_impr:+.1f}%)\n"

    # Direct comparison: at each level, how does synth+real compare to real-only?
    report += f"\n### {t('cv_impact_of_synthetic', lang)}\n\n"
    level_pairs = [
        ('exp11_real10', 'exp02_synth100_real10', '10%'),
        ('exp12_real25', 'exp03_synth100_real25', '25%'),
        ('exp13_real50', 'exp04_synth100_real50', '50%'),
        ('exp14_real75', 'exp05_synth100_real75', '75%'),
        ('exp07_real100', 'exp06_synth100_real100', '100%'),
    ]
    for version in versions:
        sc = get_synth_config(version)
        helps = 0
        hurts = 0
        for real_exp, synth_exp, label in level_pairs:
            real_m = all_results[first_v].get(real_exp, {})
            synth_m = all_results[version].get(synth_exp, {})
            rv = real_m.get('mAP', 0) if real_m.get('status') == 'completed' else 0
            sv = synth_m.get('mAP', 0) if synth_m.get('status') == 'completed' else 0
            if sv > rv and rv > 0:
                helps += 1
            elif sv < rv and rv > 0:
                hurts += 1
        if helps > hurts:
            report += f"- **{sc['name']}**: {t('cv_impact_positive', lang)} ({helps}/{helps+hurts})\n"
        elif hurts > helps:
            report += f"- **{sc['name']}**: {t('cv_impact_negative', lang)} ({hurts}/{helps+hurts})\n"
        else:
            report += f"- **{sc['name']}**: Mixed ({helps} positive, {hurts} negative)\n"

    report += "\n---\n\n"

    # =========================================================================
    # 6. DOMAIN GAP ANALYSIS
    # =========================================================================
    report += f"## 5. {t('cv_domain_gap_analysis', lang)}\n\n"
    report += f"{t('cv_domain_gap_desc', lang)}\n\n"

    report += f"| {t('cv_version', lang)} | {t('cv_synth_only_map', lang)} | {t('cv_real_only_map', lang)} | {t('cv_gap_absolute', lang)} | {t('cv_gap_relative', lang)} | {t('cv_recovery_ratio', lang)} |\n"
    report += "|---|---|---|---|---|---|\n"

    gaps = insights['domain_gaps']
    min_gap_v = None
    min_gap_pct = float('inf')
    for version in versions:
        g = gaps[version]
        sc = get_synth_config(version)
        if 0 < g['gap_pct'] < min_gap_pct:
            min_gap_pct = g['gap_pct']
            min_gap_v = version
        report += f"| {sc['name']} | {g['synth_map']:.4f} | {g['real_map']:.4f} | {g['gap_abs']:.4f} | {g['gap_pct']:.1f}% | {g['recovery_ratio']:.1f}% |\n"

    report += "\n![Domain Gap Analysis](cross_version_domain_gap.png)\n\n"

    # Dynamic conclusion
    report += f"### {t('cv_key_insight', lang)}\n\n"
    if min_gap_v:
        sc = get_synth_config(min_gap_v)
        report += f"**{sc['name']}** ({sc['num_classes']} {t('cv_classes', lang).lower()}) "
        if lang == 'es':
            report += f"tiene el menor domain gap ({min_gap_pct:.1f}%), lo que indica una mayor calidad y alineación de los datos sintéticos con los reales. "
            if sc['num_classes'] == 8:
                report += "La alineación completa de las 8 clases contribuye a este resultado."
            else:
                report += f"A pesar de tener solo {sc['num_classes']} clases, logra un domain gap competitivo."
        else:
            report += f"has the smallest domain gap ({min_gap_pct:.1f}%), indicating higher synthetic data quality and alignment with real data. "
            if sc['num_classes'] == 8:
                report += "Full 8-class alignment contributes to this result."
            else:
                report += f"Despite having only {sc['num_classes']} classes, it achieves a competitive domain gap."
    report += "\n\n---\n\n"

    # =========================================================================
    # 7. PER-CLASS ANALYSIS
    # =========================================================================
    report += f"## 6. {t('cv_per_class_analysis', lang)}\n\n"
    report += "![Per-Class Heatmap](cross_version_per_class_heatmap.png)\n\n"

    report += f"### {t('cv_per_class_champion', lang)}\n\n"
    report += f"| {t('class', lang)} | {t('cv_best', lang)} {t('cv_version', lang)} | AP |\n|---|---|---|\n"
    for cls_name, info in insights['best_per_class'].items():
        if info['version']:
            v_name = get_synth_config(info['version'])['name']
            is_minority = " *" if cls_name in MINORITY_CLASSES else ""
            report += f"| {cls_name}{is_minority} | {v_name} | {info['ap']:.4f} |\n"
        else:
            report += f"| {cls_name} | N/A | N/A |\n"

    if any(cls in MINORITY_CLASSES for cls in CLASS_NAMES):
        if lang == 'es':
            report += "\n\\* *Clase minoritaria (<2% de anotaciones)*\n"
        else:
            report += "\n\\* *Minority class (<2% of annotations)*\n"

    # Version strengths/weaknesses
    report += f"\n### {t('cv_version_strengths', lang)} / {t('cv_version_weaknesses', lang)}\n\n"
    for version in versions:
        sc = get_synth_config(version)
        strengths = [cls for cls, info in insights['best_per_class'].items() if info['version'] == version]
        weaknesses = [cls for cls, info in insights['best_per_class'].items() if info['version'] != version and info['version'] is not None]
        report += f"**{sc['name']}**:\n"
        if strengths:
            report += f"- {t('cv_version_strengths', lang)}: {', '.join(strengths)}\n"
        if weaknesses:
            report += f"- {t('cv_version_weaknesses', lang)}: {', '.join(weaknesses)}\n"
        report += "\n"

    report += "---\n\n"

    # =========================================================================
    # 8. MINORITY CLASS ANALYSIS
    # =========================================================================
    report += f"## 7. {t('cv_minority_analysis', lang)}\n\n"
    report += "![Minority Class Analysis](cross_version_minority_analysis.png)\n\n"

    for cls_name in MINORITY_CLASSES:
        report += f"### {cls_name}\n\n"
        report += f"| {t('experiment', lang)} | "
        for version in versions:
            report += f"{get_synth_config(version)['name']} | "
        report += f"{t('cv_best', lang)} |\n|---|" + "---|" * len(versions) + "---|\n"

        for exp_name in exp_names:
            report += f"| {exp_name} | "
            maps = {}
            for version in versions:
                m = all_results[version].get(exp_name, {})
                val = m.get(f'AP_{cls_name}', 0) if m.get('status') == 'completed' else 0
                maps[version] = val if val else 0

            best_val = max(maps.values()) if maps else 0
            for version in versions:
                val = maps[version]
                if val > 0 and val == best_val:
                    report += f"**{val:.4f}** | "
                elif val > 0:
                    report += f"{val:.4f} | "
                else:
                    report += "N/A | "

            if best_val > 0:
                best_v = max(maps, key=maps.get)
                report += f"{get_synth_config(best_v)['name']} |\n"
            else:
                report += "N/A |\n"

        report += "\n"

    report += "---\n\n"

    # =========================================================================
    # 9. RADAR CHART
    # =========================================================================
    report += f"## 8. {t('cv_radar_chart', lang)}\n\n"
    report += "![Radar Chart](cross_version_radar.png)\n\n"

    metric_keys = ['mAP', 'mAP_50', 'mAP_75', 'AP_small', 'AP_medium', 'AP_large']
    metric_labels = ['mAP', 'mAP@50', 'mAP@75', 'AP Small', 'AP Medium', 'AP Large']

    report += f"| {t('metric', lang)} | "
    for version in versions:
        report += f"{get_synth_config(version)['name']} | "
    report += f"{t('cv_best', lang)} |\n|---|" + "---|" * len(versions) + "---|\n"

    best_exp = 'exp09_real_synth_prop'
    for key, label in zip(metric_keys, metric_labels):
        report += f"| {label} | "
        vals = {}
        for version in versions:
            m = all_results[version].get(best_exp, {})
            if m.get('status') != 'completed':
                m = all_results[version].get('exp06_synth100_real100', {})
            val = m.get(key, 0) or 0
            vals[version] = val
            report += f"{val:.4f} | " if val > 0 else "N/A | "

        best_val = max(vals.values()) if vals else 0
        if best_val > 0:
            best_v = max(vals, key=vals.get)
            report += f"{get_synth_config(best_v)['name']} |\n"
        else:
            report += "N/A |\n"

    report += "\n---\n\n"

    # =========================================================================
    # 10. SIZE ANALYSIS
    # =========================================================================
    report += f"## 9. {t('cv_size_analysis', lang)}\n\n"
    report += "![Size Analysis](cross_version_size_analysis.png)\n\n"

    report += f"| {t('cv_version', lang)} | {t('cv_ap_small', lang)} | {t('cv_ap_medium', lang)} | {t('cv_ap_large', lang)} |\n"
    report += "|---|---|---|---|\n"
    for version in versions:
        sc = get_synth_config(version)
        sa = insights['size_analysis'].get(version, {})
        report += f"| {sc['name']} | {sa.get('AP_small', 0):.4f} | {sa.get('AP_medium', 0):.4f} | {sa.get('AP_large', 0):.4f} |\n"

    report += "\n---\n\n"

    # =========================================================================
    # 11. TRAINING EFFICIENCY
    # =========================================================================
    report += f"## 10. {t('cv_training_efficiency', lang)}\n\n"
    report += "![Training Efficiency](cross_version_training_efficiency.png)\n\n"

    report += f"| {t('cv_version', lang)} | {t('cv_training_hours', lang)} | {t('cv_avg_map', lang)} | {t('cv_performance_ratio', lang)} |\n"
    report += "|---|---|---|---|\n"
    for version in versions:
        sc = get_synth_config(version)
        avg_h = insights['training_efficiency'].get(version, 0)
        avg_m = insights['avg_maps'].get(version, 0)
        ratio = (avg_m / avg_h) if avg_h > 0 else 0
        report += f"| {sc['name']} | {avg_h:.1f} | {avg_m:.4f} | {ratio:.4f} |\n"

    report += "\n---\n\n"

    # =========================================================================
    # 12. SYNTH BENEFIT
    # =========================================================================
    report += f"## 11. {t('cv_synth_benefit', lang)}\n\n"
    report += f"{t('cv_synth_benefit_desc', lang)}.\n\n"
    report += "![Synth Benefit](cross_version_synth_benefit.png)\n\n"

    report += f"| {t('real_data_pct', lang)} | "
    for version in versions:
        report += f"{get_synth_config(version)['name']} Δ mAP | "
    report += "\n|---|" + "---|" * len(versions) + "\n"

    ratio_labels = {0.1: '10%', 0.25: '25%', 0.5: '50%', 0.75: '75%', 1.0: '100%'}
    for ratio, label in ratio_labels.items():
        report += f"| {label} | "
        for version in versions:
            delta = insights['synth_benefit'].get(version, {}).get(ratio, 0)
            if delta != 0:
                report += f"{delta:+.4f} | "
            else:
                report += "N/A | "
        report += "\n"

    report += "\n---\n\n"

    # =========================================================================
    # 12.5 PAIRED COMPARISON: Real-Only vs Synth+Real
    # =========================================================================
    report += f"## 12. {t('cv_paired_comparison', lang)}\n\n"
    report += f"{t('cv_paired_desc', lang)}\n\n"
    report += "![Paired Comparison](cross_version_paired_comparison.png)\n\n"

    # Paired comparison table
    pairs_for_table = [
        ('exp11_real10', 'exp02_synth100_real10', '10%'),
        ('exp12_real25', 'exp03_synth100_real25', '25%'),
        ('exp13_real50', 'exp04_synth100_real50', '50%'),
        ('exp14_real75', 'exp05_synth100_real75', '75%'),
        ('exp07_real100', 'exp06_synth100_real100', '100%'),
    ]

    report += f"| {t('real_data_pct', lang)} | {t('cv_real_only_baseline', lang)} | "
    for version in versions:
        sc = get_synth_config(version)
        report += f"{sc['name']} (Synth+Real) | {t('cv_delta', lang)} | "
    report += "\n|---|---|"
    for _ in versions:
        report += "---|---|"
    report += "\n"

    for real_exp, synth_exp, label in pairs_for_table:
        real_m = all_results[versions[0]].get(real_exp, {})
        real_val = real_m.get('mAP', 0) if real_m.get('status') == 'completed' else 0
        report += f"| {label} | {real_val:.4f} | "

        for version in versions:
            synth_m = all_results[version].get(synth_exp, {})
            synth_val = synth_m.get('mAP', 0) if synth_m.get('status') == 'completed' else 0
            delta = synth_val - real_val if (synth_val > 0 and real_val > 0) else 0
            delta_str = f"**{delta:+.4f}**" if delta > 0 else f"{delta:+.4f}"
            report += f"{synth_val:.4f} | {delta_str} | "
        report += "\n"

    # Dynamic paired comparison conclusion
    report += f"\n### {t('cv_paired_conclusion', lang)}\n\n"
    for version in versions:
        sc = get_synth_config(version)
        positive_levels = 0
        negative_levels = 0
        for real_exp, synth_exp, label in pairs_for_table:
            real_m = all_results[versions[0]].get(real_exp, {})
            synth_m = all_results[version].get(synth_exp, {})
            real_val = real_m.get('mAP', 0) if real_m.get('status') == 'completed' else 0
            synth_val = synth_m.get('mAP', 0) if synth_m.get('status') == 'completed' else 0
            if synth_val > 0 and real_val > 0:
                if synth_val > real_val:
                    positive_levels += 1
                else:
                    negative_levels += 1

        report += f"- **{sc['name']}**: "
        if positive_levels > negative_levels:
            if lang == 'es':
                report += f"Impacto positivo en {positive_levels} de {positive_levels + negative_levels} niveles. "
                report += "Añadir estos datos sintéticos mejora consistentemente el rendimiento.\n"
            else:
                report += f"Positive impact at {positive_levels} of {positive_levels + negative_levels} levels. "
                report += "Adding this synthetic data consistently improves performance.\n"
        elif negative_levels > positive_levels:
            if lang == 'es':
                report += f"Impacto negativo en {negative_levels} de {positive_levels + negative_levels} niveles. "
                report += "El domain gap de estos datos sintéticos supera su beneficio en la mayoría de casos.\n"
            else:
                report += f"Negative impact at {negative_levels} of {positive_levels + negative_levels} levels. "
                report += "The domain gap of this synthetic data outweighs its benefit in most cases.\n"
        else:
            if lang == 'es':
                report += f"Resultados mixtos ({positive_levels} positivos, {negative_levels} negativos).\n"
            else:
                report += f"Mixed results ({positive_levels} positive, {negative_levels} negative).\n"

    report += "\n---\n\n"

    # =========================================================================
    # 13. RANKING
    # =========================================================================
    report += f"## 13. {t('cv_ranking', lang)}\n\n"
    report += "![Version Ranking](version_ranking_heatmap.png)\n\n"

    report += f"### {t('cv_wins_by_version', lang)}\n\n"
    report += f"| {t('cv_version', lang)} | {t('cv_wins_by_version', lang)} | {t('cv_avg_map', lang)} | {t('cv_completed_exps', lang)} |\n"
    report += "|---|---|---|---|\n"
    for version, wins in sorted(insights['win_counts'].items(), key=lambda x: -x[1]):
        sc = get_synth_config(version)
        completed = sum(1 for e in exp_names if all_results[version].get(e, {}).get('status') == 'completed')
        report += f"| {sc['name']} | **{wins}** | {insights['avg_maps'].get(version, 0):.4f} | {completed} |\n"

    report += "\n---\n\n"

    # =========================================================================
    # 14. CONCLUSIONS
    # =========================================================================
    report += f"## 14. {t('cv_conclusions', lang)}\n\n"

    # Hypothesis: more classes = better performance
    report += f"### {t('cv_hypothesis_classes', lang)}\n\n"
    sorted_by_avg = sorted(versions, key=lambda v: -insights['avg_maps'].get(v, 0))
    sorted_by_classes = sorted(versions, key=lambda v: -get_synth_config(v)['num_classes'])

    if sorted_by_avg == sorted_by_classes:
        if lang == 'es':
            report += "**Confirmada.** El ranking de rendimiento coincide con el número de clases: "
        else:
            report += "**Confirmed.** Performance ranking matches number of classes: "
        report += ", ".join(f"{get_synth_config(v)['name']} ({get_synth_config(v)['num_classes']} cls, mAP={insights['avg_maps'][v]:.4f})" for v in sorted_by_avg)
        report += ".\n\n"
    else:
        if lang == 'es':
            report += "**Parcialmente confirmada.** El ranking no coincide exactamente:\n"
        else:
            report += "**Partially confirmed.** Ranking does not exactly match:\n"
        cls_ranking = ', '.join(
            f"{get_synth_config(v)['name']} ({get_synth_config(v)['num_classes']})"
            for v in sorted_by_classes
        )
        map_ranking = ', '.join(
            f"{get_synth_config(v)['name']} ({insights['avg_maps'][v]:.4f})"
            for v in sorted_by_avg
        )
        report += f"- By classes: {cls_ranking}\n"
        report += f"- By mAP: {map_ranking}\n\n"

    # Hypothesis: full alignment reduces domain gap
    report += f"### {t('cv_hypothesis_alignment', lang)}\n\n"
    sorted_by_gap = sorted([v for v in versions if gaps[v]['gap_pct'] > 0], key=lambda v: gaps[v]['gap_pct'])
    if sorted_by_gap:
        best_gap_v = sorted_by_gap[0]
        best_gap_cls = get_synth_config(best_gap_v)['num_classes']
        if best_gap_cls == 8:
            if lang == 'es':
                report += f"**Confirmada.** {get_synth_config(best_gap_v)['name']} (alineación completa, 8 clases) tiene el menor domain gap ({gaps[best_gap_v]['gap_pct']:.1f}%).\n\n"
            else:
                report += f"**Confirmed.** {get_synth_config(best_gap_v)['name']} (full alignment, 8 classes) has the smallest domain gap ({gaps[best_gap_v]['gap_pct']:.1f}%).\n\n"
        else:
            if lang == 'es':
                report += f"**No confirmada.** {get_synth_config(best_gap_v)['name']} ({best_gap_cls} clases) tiene menor domain gap que versiones con más clases.\n\n"
            else:
                report += f"**Not confirmed.** {get_synth_config(best_gap_v)['name']} ({best_gap_cls} classes) has lower domain gap than versions with more classes.\n\n"

    # Key findings summary
    report += f"### {t('key_findings', lang)}\n\n"
    if insights['is_dominant']:
        sc = get_synth_config(dominant)
        if lang == 'es':
            report += f"1. **{sc['name']}** es la versión dominante, ganando en {insights['win_counts'][dominant]} de {len(exp_names)} experimentos con un margen promedio de {insights['margin_of_victory']:.4f} mAP.\n"
        else:
            report += f"1. **{sc['name']}** is the dominant version, winning {insights['win_counts'][dominant]} of {len(exp_names)} experiments with an average margin of {insights['margin_of_victory']:.4f} mAP.\n"
    else:
        if lang == 'es':
            report += f"1. No hay una versión claramente dominante. Las versiones muestran rendimientos competitivos.\n"
        else:
            report += f"1. No clearly dominant version. Versions show competitive performance.\n"

    # Trade-offs
    for i, trade_off in enumerate(insights.get('trade_offs', []), 2):
        report += f"{i}. {t('cv_trade_off', lang)}: {trade_off}\n"

    # Surprising findings
    for i, finding in enumerate(insights.get('surprising_findings', [])[:3], len(insights.get('trade_offs', [])) + 2):
        report += f"{i}. {t('cv_surprising_finding', lang)}: {finding}\n"

    report += "\n---\n\n"

    # =========================================================================
    # 15. RECOMMENDATIONS
    # =========================================================================
    report += f"## 15. {t('cv_recommendations', lang)}\n\n"

    # Production recommendation
    best_avg_v = max(versions, key=lambda v: insights['avg_maps'].get(v, 0))
    best_sc = get_synth_config(best_avg_v)
    report += f"### {t('production_deployment', lang)}\n\n"
    report += f"- {t('cv_recommendation_production', lang)} **{best_sc['name']}** "
    if lang == 'es':
        report += f"(mAP promedio: {insights['avg_maps'][best_avg_v]:.4f}, {best_sc['num_classes']} clases).\n"
    else:
        report += f"(avg mAP: {insights['avg_maps'][best_avg_v]:.4f}, {best_sc['num_classes']} classes).\n"

    # Minority class recommendation
    report += f"\n### {t('cv_recommendation_minority', lang).rstrip(', usar').rstrip(', use')}\n\n"
    for cls_name, best_v in insights.get('minority_best', {}).items():
        if best_v:
            report += f"- {t('cv_best_for_class', lang)} **{cls_name}**: {get_synth_config(best_v)['name']}\n"

    # Future work
    report += f"\n### {t('future_work', lang)}\n\n"
    report += f"1. {t('cv_future_more_versions', lang)}\n"
    report += f"2. {t('cv_future_class_alignment', lang)}\n"
    report += f"3. {t('cv_future_ensemble', lang)}\n"
    report += f"4. {t('cv_future_domain_adapt', lang)}\n"

    report += "\n---\n\n"

    # =========================================================================
    # 16. FILES REFERENCE
    # =========================================================================
    report += f"## 16. {t('cv_files_reference', lang)}\n\n"
    report += "| File | Description |\n|---|---|\n"
    report += "| `cross_version_comparison.png` | mAP bar chart by experiment and version |\n"
    report += "| `cross_version_map_curves.png` | mAP vs real data ratio curves |\n"
    report += "| `cross_version_domain_gap.png` | Domain gap and recovery analysis |\n"
    report += "| `cross_version_per_class_heatmap.png` | Per-class AP heatmap |\n"
    report += "| `cross_version_minority_analysis.png` | Minority class comparison |\n"
    report += "| `cross_version_radar.png` | Multi-metric radar chart |\n"
    report += "| `cross_version_size_analysis.png` | AP by object size |\n"
    report += "| `cross_version_training_efficiency.png` | Training duration comparison |\n"
    report += "| `cross_version_synth_benefit.png` | Synthetic data benefit curves |\n"
    report += "| `cross_version_paired_comparison.png` | Paired comparison: Real-Only vs Synth+Real |\n"
    report += "| `version_ranking_heatmap.png` | Version ranking heatmap |\n"
    report += f"| `CROSS_VERSION_REPORT.md` | English report |\n"
    report += f"| `CROSS_VERSION_REPORT_ES.md` | Spanish report |\n"

    report += f"\n---\n\n*{t('cv_report_auto_gen', lang)}*\n"

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Saved: {output_path}")


def generate_cross_version_report(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Generate markdown report comparing results across synthetic dataset versions.

    Args:
        all_results: Dict mapping version to experiment metrics
        output_path: Path to save the report
    """
    versions = list(all_results.keys())

    report = f"""# Cross-Version Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report compares experiment results across different synthetic dataset versions.

### Versions Compared

| Version | Name | Description |
|---------|------|-------------|
"""

    for version in versions:
        synth_config = get_synth_config(version)
        report += f"| {version} | {synth_config['name']} | {synth_config['description']} |\n"

    # Comparison table
    report += """

## mAP Comparison Table

| Experiment | """

    for version in versions:
        synth_config = get_synth_config(version)
        report += f"{synth_config['name']} | "

    report += "Best Version |\n|---|"
    for _ in versions:
        report += "---|"
    report += "---|\n"

    exp_names = [exp['name'] for exp in EXPERIMENTS]
    version_wins = {v: 0 for v in versions}

    for exp_name in exp_names:
        report += f"| {exp_name} | "
        maps = {}
        for version in versions:
            metrics = all_results[version].get(exp_name, {})
            mAP = metrics.get('mAP', 0) if metrics.get('status') == 'completed' else 0
            maps[version] = mAP
            report += f"{mAP:.4f} | " if mAP > 0 else "N/A | "

        # Find best
        if any(m > 0 for m in maps.values()):
            best_version = max(maps, key=maps.get)
            version_wins[best_version] += 1
            synth_config = get_synth_config(best_version)
            report += f"**{synth_config['name']}** |\n"
        else:
            report += "N/A |\n"

    # Summary statistics
    report += """

## Summary Statistics

### Wins by Version

| Version | Experiments Won |
|---------|-----------------|
"""
    for version, wins in sorted(version_wins.items(), key=lambda x: -x[1]):
        synth_config = get_synth_config(version)
        report += f"| {synth_config['name']} | {wins} |\n"

    # Average mAP per version
    report += """

### Average mAP by Version

| Version | Average mAP | Completed Experiments |
|---------|-------------|----------------------|
"""
    for version in versions:
        maps = []
        for exp_name in exp_names:
            metrics = all_results[version].get(exp_name, {})
            if metrics.get('status') == 'completed' and 'mAP' in metrics:
                maps.append(metrics['mAP'])
        if maps:
            synth_config = get_synth_config(version)
            report += f"| {synth_config['name']} | {sum(maps)/len(maps):.4f} | {len(maps)} |\n"

    report += """

## Visualizations

- ![Cross-Version Comparison](cross_version_comparison.png)
- ![Version Ranking Heatmap](version_ranking_heatmap.png)

---

*Report generated automatically by collect_results.py*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Saved: {output_path}")


def run_cross_version_comparison(
    output_dir: Path,
    no_plots: bool = False,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> None:
    """
    Run complete cross-version comparison analysis.

    Args:
        output_dir: Output directory for comparison results
        no_plots: If True, skip generating plots
        batch_mode: Batch sampling mode
    """
    print("\n" + "="*70)
    print("CROSS-VERSION COMPARISON")
    print("="*70)

    # Collect results from all versions
    print("\nCollecting results from all versions...")
    all_results = collect_all_version_results(batch_mode)

    if not all_results:
        print("\nNo version results found. Run experiments first.")
        return

    print(f"\nFound results for {len(all_results)} version(s): {', '.join(all_results.keys())}")

    # Create comparison directory
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    if not no_plots and HAS_MATPLOTLIB:
        print("\nGenerating cross-version visualizations...")
        # Original plots
        plot_cross_version_comparison(all_results, comparison_dir / "cross_version_comparison.png")
        plot_version_ranking(all_results, comparison_dir / "version_ranking_heatmap.png")
        # New comprehensive plots
        plot_cross_version_map_curves(all_results, comparison_dir / "cross_version_map_curves.png")
        plot_cross_version_per_class_heatmap(all_results, comparison_dir / "cross_version_per_class_heatmap.png")
        plot_cross_version_radar(all_results, comparison_dir / "cross_version_radar.png")
        plot_cross_version_minority_analysis(all_results, comparison_dir / "cross_version_minority_analysis.png")
        plot_cross_version_training_efficiency(all_results, comparison_dir / "cross_version_training_efficiency.png")
        plot_cross_version_domain_gap(all_results, comparison_dir / "cross_version_domain_gap.png")
        plot_cross_version_size_analysis(all_results, comparison_dir / "cross_version_size_analysis.png")
        plot_cross_version_synth_benefit(all_results, comparison_dir / "cross_version_synth_benefit.png")
        plot_cross_version_paired_comparison(all_results, comparison_dir / "cross_version_paired_comparison.png")

    # Generate comprehensive reports in both languages
    print("\nGenerating cross-version reports...")
    print("  Generating English report...")
    generate_cross_version_report_full(all_results, comparison_dir / "CROSS_VERSION_REPORT.md", lang='en')
    print("  Generating Spanish report...")
    generate_cross_version_report_full(all_results, comparison_dir / "CROSS_VERSION_REPORT_ES.md", lang='es')

    print("\n" + "="*70)
    print("CROSS-VERSION COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {comparison_dir}")
    print("\nGenerated files:")
    for f in sorted(comparison_dir.iterdir()):
        print(f"  - {f.name}")


# =============================================================================
# CROSS-BATCH-MODE COMPARISON
# =============================================================================

def collect_all_batch_mode_results(
    synth_version: str = DEFAULT_SYNTH_VERSION
) -> Dict[str, Dict[str, Dict]]:
    """
    Collect results from all available batch modes for a given synthetic version.

    Args:
        synth_version: Synthetic dataset version

    Returns:
        Dict mapping batch_mode name to experiment metrics dict
    """
    all_results = {}

    for bm in BATCH_MODES:
        versioned_paths = get_versioned_output_paths(synth_version, bm)
        results_dir = Path(versioned_paths['results'])

        # Check if this batch mode has any results via state file
        state_file = results_dir / f"experiment_state_{synth_version}.json"
        # Also check for real-only state file
        real_only_state = Path(RESULTS_OUTPUT_DIR) / "experiment_state.json"

        has_state = state_file.exists() or real_only_state.exists()
        if has_state:
            print(f"\n  Loading results for batch mode: {bm}")
            metrics = collect_all_metrics(synth_version, bm)

            completed = sum(1 for m in metrics.values() if m.get('status') == 'completed')
            if completed > 0:
                all_results[bm] = metrics
                print(f"    {completed} completed experiments")
            else:
                print(f"    No completed experiments")
        else:
            print(f"  Skipping batch mode {bm}: no state file found")

    return all_results


def _bm_get_completed_exp_names(all_results):
    """Get experiment names completed in at least one batch mode."""
    batch_modes = list(all_results.keys())
    names = []
    for exp in ALL_EXPERIMENTS:
        for bm in batch_modes:
            m = all_results[bm].get(exp['name'], {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                names.append(exp['name'])
                break
    return names


def _bm_get_common_exp_names(all_results):
    """Get experiment names completed in ALL batch modes."""
    batch_modes = list(all_results.keys())
    names = []
    for exp in ALL_EXPERIMENTS:
        if all(
            all_results[bm].get(exp['name'], {}).get('status') == 'completed'
            for bm in batch_modes
        ):
            names.append(exp['name'])
    return names


def plot_cross_batch_mode_comparison(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Grouped bar chart comparing mAP across batch modes per experiment (with Real-Only baselines)."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    # Synth+real experiments
    exp_names = [exp['name'] for exp in EXPERIMENTS]
    # Real-only pair mapping
    real_only_pair = {
        'exp02_synth100_real10': 'exp11_real10',
        'exp03_synth100_real25': 'exp12_real25',
        'exp04_synth100_real50': 'exp13_real50',
        'exp05_synth100_real75': 'exp14_real75',
        'exp06_synth100_real100': 'exp07_real100',
    }

    fig, ax = plt.subplots(figsize=(max(16, len(exp_names) * 2), 8))
    x = np.arange(len(exp_names))
    n_groups = len(batch_modes) + 1  # batch modes + real-only baseline
    width = 0.8 / n_groups
    bm_colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))

    # Real-only baseline bars (use first batch mode's real-only data)
    first_bm = batch_modes[0]
    real_vals = []
    for exp_name in exp_names:
        real_pair = real_only_pair.get(exp_name)
        if real_pair:
            rm = all_results[first_bm].get(real_pair, {})
            real_vals.append(rm.get('mAP', 0) if rm.get('status') == 'completed' else 0)
        else:
            real_vals.append(0)

    offset_ro = (0 - n_groups / 2 + 0.5) * width
    bars_ro = ax.bar(x + offset_ro, real_vals, width, label='Real-Only',
                     color='#888888', edgecolor='#555555', linewidth=0.5)
    for bar, val in zip(bars_ro, real_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45, color='#555555')

    # Batch mode bars
    for i, bm in enumerate(batch_modes):
        values = []
        for exp_name in exp_names:
            m = all_results[bm].get(exp_name, {})
            values.append(m.get('mAP', 0) if m.get('status') == 'completed' else 0)
        offset = (i + 1 - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=f'Synth+Real ({bm})',
                      color=bm_colors[i], edgecolor=bm_colors[i] * 0.8, linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('mAP@0.5:0.95')
    ax.set_title('Batch Mode Comparison: mAP by Experiment\n(with Real-Only baselines)')
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace('exp', 'E') for e in exp_names], rotation=45, ha='right')
    ax.legend(title='Batch Mode', loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_ranking(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Heatmap showing which batch mode performs best per experiment."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    exp_names = _bm_get_common_exp_names(all_results)
    if len(batch_modes) < 2 or not exp_names:
        return

    mAP_matrix = []
    for exp_name in exp_names:
        row = []
        for bm in batch_modes:
            m = all_results[bm].get(exp_name, {})
            row.append(m.get('mAP', 0) if m.get('status') == 'completed' else 0)
        mAP_matrix.append(row)
    mAP_matrix = np.array(mAP_matrix)

    fig, ax = plt.subplots(figsize=(max(8, len(batch_modes) * 3), max(8, len(exp_names) * 0.5)))
    im = ax.imshow(mAP_matrix, cmap='YlGn', aspect='auto')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('mAP@0.5:0.95', rotation=-90, va='bottom')

    ax.set_xticks(np.arange(len(batch_modes)))
    ax.set_yticks(np.arange(len(exp_names)))
    ax.set_xticklabels(batch_modes, fontsize=11)
    ax.set_yticklabels(exp_names, fontsize=9)

    for i in range(len(exp_names)):
        for j in range(len(batch_modes)):
            val = mAP_matrix[i, j]
            if val == mAP_matrix[i].max() and val > 0:
                ax.text(j, i, f'{val:.4f}*', ha='center', va='center',
                        color='darkgreen', fontweight='bold', fontsize=9)
            elif val > 0:
                ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=9)

    ax.set_title('Batch Mode Ranking by Experiment\n(* indicates best mode for that experiment)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_map_curves(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """mAP vs real data ratio curves, one line per batch mode, plus Real-Only baseline."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    first_bm = batch_modes[0]

    fig, ax = plt.subplots(figsize=(12, 7))

    synth_real_exps = [e for e in EXPERIMENTS if e['id'] <= 7]
    ratios = [e['real_ratio'] for e in synth_real_exps]
    exp_names = [e['name'] for e in synth_real_exps]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))
    markers = ['o', 's', '^', 'D', 'v']

    for i, bm in enumerate(batch_modes):
        maps = []
        for en in exp_names:
            m = all_results[bm].get(en, {})
            maps.append(m.get('mAP', 0) if m.get('status') == 'completed' else 0)
        marker = markers[i % len(markers)]
        ax.plot([r * 100 for r in ratios], maps, f'-{marker}', linewidth=2.5,
                markersize=9, label=f'Synth+Real ({bm})', color=colors[i])
        for r, v in zip(ratios, maps):
            if v > 0:
                ax.annotate(f'{v:.3f}', (r * 100, v), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=7, color=colors[i])

    # Real-only baseline
    real_only_exps = [
        ('exp11_real10', 0.1), ('exp12_real25', 0.25),
        ('exp13_real50', 0.5), ('exp14_real75', 0.75),
        ('exp07_real100', 1.0),
    ]
    real_r, real_m = [], []
    for en, ratio in real_only_exps:
        m = all_results[first_bm].get(en, {})
        val = m.get('mAP', 0) if m.get('status') == 'completed' else 0
        if val > 0:
            real_r.append(ratio * 100)
            real_m.append(val)
    if real_m:
        ax.plot(real_r, real_m, '--D', linewidth=2.5, markersize=9,
                label='Real-Only (no synthetic)', color='#555555')
        for r, v in zip(real_r, real_m):
            ax.annotate(f'{v:.3f}', (r, v), textcoords="offset points",
                        xytext=(0, -15), ha='center', fontsize=7, color='#555555')

    ax.set_xlabel('Percentage of Real Data (%)', fontsize=12)
    ax.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax.set_title('Batch Mode: mAP vs Real Data Ratio\n(Batch modes vs Real-Only baseline)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-5, 105)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_batch_mode_per_class(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Heatmap of per-class AP across batch modes for key experiments."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    key_exps = ['exp01_synth100', 'exp07_real100', 'exp06_synth100_real100',
                'exp08_real_synth_minor', 'exp09_real_synth_prop']
    key_labels = ['Synth 100%', 'Real 100%', 'Synth+Real 100%', 'Minority Strategy', 'Proportional Strategy']

    col_labels = []
    col_exp_bm = []
    for exp_name, exp_label in zip(key_exps, key_labels):
        if exp_name == 'exp07_real100':
            col_labels.append(f"{exp_label}\n(Baseline)")
            col_exp_bm.append((exp_name, batch_modes[0]))
        else:
            for bm in batch_modes:
                col_labels.append(f"{exp_label}\n{bm}")
                col_exp_bm.append((exp_name, bm))

    data = []
    for cls_name in CLASS_NAMES:
        row = []
        for exp_name, bm in col_exp_bm:
            m = all_results[bm].get(exp_name, {})
            ap = m.get(f'AP_{cls_name}', 0) if m.get('status') == 'completed' else 0
            row.append(ap if ap else 0)
        data.append(row)
    data = np.array(data)

    if data.max() == 0:
        print(f"  Skipped {output_path.name}: No per-class AP data")
        return

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 1.2), 8))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('AP@0.5:0.95', rotation=-90, va='bottom')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(len(CLASS_NAMES)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if val > 0:
                color = 'white' if val > data.max() * 0.65 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6, color=color)

    ax.set_title('Per-Class AP Across Batch Modes and Key Experiments', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_per_class_delta(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Heatmap showing per-class AP difference between batch modes."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    if len(batch_modes) < 2:
        return

    exp_names = _bm_get_common_exp_names(all_results)
    if not exp_names:
        return

    baseline = batch_modes[0]
    compare = batch_modes[1]

    matrix = np.full((len(exp_names), len(CLASS_NAMES)), np.nan)
    for i, exp_name in enumerate(exp_names):
        base_m = all_results[baseline].get(exp_name, {})
        comp_m = all_results[compare].get(exp_name, {})
        for j, cls in enumerate(CLASS_NAMES):
            base_ap = base_m.get(f'AP_{cls}')
            comp_ap = comp_m.get(f'AP_{cls}')
            if base_ap is not None and comp_ap is not None:
                matrix[i, j] = comp_ap - base_ap

    fig, ax = plt.subplots(figsize=(max(12, len(CLASS_NAMES) * 0.8), max(6, len(exp_names) * 0.5)))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels([n.replace('exp', 'E') for n in exp_names], fontsize=8)

    for i in range(len(exp_names)):
        for j in range(len(CLASS_NAMES)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'black' if abs(val) < vmax * 0.5 else 'white'
                ax.text(j, i, f'{val:+.3f}', ha='center', va='center', fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label=f'AP delta ({compare} - {baseline})')
    ax.set_title(f'Per-Class AP Difference: {compare} vs {baseline}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_radar(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Radar chart comparing batch modes on multiple metrics for the best experiment."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    best_exp = 'exp09_real_synth_prop'
    metric_keys = ['mAP', 'mAP_50', 'mAP_75', 'AP_small', 'AP_medium', 'AP_large']
    metric_labels = ['mAP', 'mAP@50', 'mAP@75', 'AP Small', 'AP Medium', 'AP Large']
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))

    bm_data = {}
    has_data = False
    for bm in batch_modes:
        m = all_results[bm].get(best_exp, {})
        if m.get('status') != 'completed':
            m = all_results[bm].get('exp06_synth100_real100', {})
        values = [m.get(k, 0) or 0 for k in metric_keys]
        bm_data[bm] = values
        if any(v > 0 for v in values):
            has_data = True

    if not has_data:
        print(f"  Skipped {output_path.name}: No multi-metric data")
        return

    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, bm in enumerate(batch_modes):
        values = bm_data[bm] + bm_data[bm][:1]
        ax.plot(angles, values, '-o', linewidth=2, markersize=6, label=bm, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_title(f'Multi-Metric Comparison by Batch Mode ({best_exp})', fontsize=13, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_minority_analysis(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Bar chart comparing minority class AP across batch modes for key experiments."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    key_exps = ['exp01_synth100', 'exp06_synth100_real100', 'exp07_real100',
                'exp08_real_synth_minor', 'exp09_real_synth_prop']
    key_labels = ['Synth 100%', 'Synth+Real', 'Real 100%', 'Minority', 'Proportional']
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))

    minority_classes = MINORITY_CLASSES if MINORITY_CLASSES else ['Can', 'Metal_Debris']
    fig, axes = plt.subplots(1, len(minority_classes), figsize=(8 * len(minority_classes), 7))
    if len(minority_classes) == 1:
        axes = [axes]

    for cls_idx, (cls_name, ax) in enumerate(zip(minority_classes, axes)):
        x = np.arange(len(key_exps))
        width = 0.8 / max(len(batch_modes), 1)

        for i, bm in enumerate(batch_modes):
            ap_vals = []
            for exp_name in key_exps:
                m = all_results[bm].get(exp_name, {})
                ap = m.get(f'AP_{cls_name}', 0) if m.get('status') == 'completed' else 0
                ap_vals.append(ap if ap else 0)
            offset = (i - len(batch_modes) / 2 + 0.5) * width
            bars = ax.bar(x + offset, ap_vals, width, label=bm, color=colors[i])
            for bar, val in zip(bars, ap_vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_title(f'{cls_name} (Minority Class)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(key_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('AP@0.5:0.95')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Minority Class AP Across Batch Modes', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_training_efficiency(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Bar chart comparing training duration across batch modes."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    exp_names = _bm_get_common_exp_names(all_results)
    if not exp_names:
        return

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))
    fig, ax = plt.subplots(figsize=(max(14, len(exp_names) * 1.5), 7))
    x = np.arange(len(exp_names))
    width = 0.8 / max(len(batch_modes), 1)
    has_data = False

    for i, bm in enumerate(batch_modes):
        durations = []
        for exp_name in exp_names:
            m = all_results[bm].get(exp_name, {})
            training = m.get('training', {})
            dur_s = training.get('duration_seconds', 0) or 0 if isinstance(training, dict) else 0
            durations.append(dur_s / 3600)
            if dur_s > 0:
                has_data = True
        offset = (i - len(batch_modes) / 2 + 0.5) * width
        ax.bar(x + offset, durations, width, label=bm, color=colors[i])

    if not has_data:
        plt.close()
        print(f"  Skipped {output_path.name}: No training duration data")
        return

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Training Duration (hours)')
    ax.set_title('Training Duration Comparison Across Batch Modes')
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace('exp', 'E') for e in exp_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_domain_gap(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Domain gap and recovery ratio per batch mode."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Synth-only vs Real-only per batch mode
    x = np.arange(len(batch_modes))
    synth_maps, real_maps = [], []
    for bm in batch_modes:
        sm = all_results[bm].get('exp01_synth100', {})
        rm = all_results[bm].get('exp07_real100', {})
        synth_maps.append(sm.get('mAP', 0) if sm.get('status') == 'completed' else 0)
        real_maps.append(rm.get('mAP', 0) if rm.get('status') == 'completed' else 0)

    width = 0.3
    ax1.bar(x - width / 2, synth_maps, width, label='Synth-Only (exp01)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width / 2, real_maps, width, label='Real-Only (exp07)', color='#27ae60', alpha=0.8)
    for i, (s, r) in enumerate(zip(synth_maps, real_maps)):
        if s > 0 and r > 0:
            ax1.annotate(f'Gap: {r - s:.3f}', (i, max(s, r) + 0.01),
                         ha='center', fontsize=9, fontweight='bold', color='#e67e22')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_modes)
    ax1.set_ylabel('mAP@0.5:0.95')
    ax1.set_title('Domain Gap: Synth-Only vs Real-Only')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right: Recovery ratio per batch mode
    recovery = []
    for bm in batch_modes:
        cm = all_results[bm].get('exp06_synth100_real100', {})
        rm = all_results[bm].get('exp07_real100', {})
        c_map = cm.get('mAP', 0) if cm.get('status') == 'completed' else 0
        r_map = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
        recovery.append((c_map / r_map * 100) if r_map > 0 else 0)

    bars = ax2.bar(x, recovery, 0.5, color=colors[:len(batch_modes)])
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='100% Recovery')
    for bar, val in zip(bars, recovery):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_modes)
    ax2.set_ylabel('Recovery Ratio (%)')
    ax2.set_title('Performance Recovery: Combined / Real-Only')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Domain Gap Analysis Across Batch Modes', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_size_analysis(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """AP by object size across batch modes for the best experiment."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    best_exp = 'exp09_real_synth_prop'
    size_metrics = ['AP_small', 'AP_medium', 'AP_large']
    size_labels = ['Small (<32^2)', 'Medium (32^2-96^2)', 'Large (>96^2)']
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(size_metrics))
    width = 0.8 / max(len(batch_modes), 1)
    has_data = False

    for i, bm in enumerate(batch_modes):
        m = all_results[bm].get(best_exp, {})
        if m.get('status') != 'completed':
            m = all_results[bm].get('exp06_synth100_real100', {})
        values = [m.get(k, 0) or 0 for k in size_metrics]
        if any(v > 0 for v in values):
            has_data = True
        offset = (i - len(batch_modes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=bm, color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    if not has_data:
        plt.close()
        print(f"  Skipped {output_path.name}: No size-based AP data")
        return

    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.set_ylabel('AP@0.5:0.95')
    ax.set_title(f'AP by Object Size Across Batch Modes ({best_exp})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_synth_benefit(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Synthetic data benefit at each real ratio, per batch mode."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    pairs = [
        ('exp02_synth100_real10', 'exp11_real10', '10%'),
        ('exp03_synth100_real25', 'exp12_real25', '25%'),
        ('exp04_synth100_real50', 'exp13_real50', '50%'),
        ('exp05_synth100_real75', 'exp14_real75', '75%'),
        ('exp06_synth100_real100', 'exp07_real100', '100%'),
    ]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))
    markers = ['o', 's', '^', 'D', 'v']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Delta mAP
    for i, bm in enumerate(batch_modes):
        deltas = []
        for synth_exp, real_exp, _ in pairs:
            sm = all_results[bm].get(synth_exp, {})
            rm = all_results[bm].get(real_exp, {})
            s_map = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r_map = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            deltas.append(s_map - r_map if (s_map > 0 and r_map > 0) else 0)
        marker = markers[i % len(markers)]
        ax1.plot(range(len(pairs)), deltas, f'-{marker}', linewidth=2.5,
                 markersize=9, label=bm, color=colors[i])
        for j, d in enumerate(deltas):
            if d != 0:
                ax1.annotate(f'{d:+.3f}', (j, d), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=8, color=colors[i])

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xticks(range(len(pairs)))
    ax1.set_xticklabels([p[2] for p in pairs])
    ax1.set_xlabel('Real Data %')
    ax1.set_ylabel('Delta mAP (Synth+Real minus Real-Only)')
    ax1.set_title('Synthetic Data Benefit by Batch Mode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Relative improvement %
    for i, bm in enumerate(batch_modes):
        improvements = []
        for synth_exp, real_exp, _ in pairs:
            sm = all_results[bm].get(synth_exp, {})
            rm = all_results[bm].get(real_exp, {})
            s_map = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r_map = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            improvements.append(((s_map - r_map) / r_map * 100) if r_map > 0 and s_map > 0 else 0)
        marker = markers[i % len(markers)]
        ax2.plot(range(len(pairs)), improvements, f'-{marker}', linewidth=2.5,
                 markersize=9, label=bm, color=colors[i])

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xticks(range(len(pairs)))
    ax2.set_xticklabels([p[2] for p in pairs])
    ax2.set_xlabel('Real Data %')
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.set_title('Relative Benefit of Synthetic Data (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Synthetic Data Benefit Analysis Across Batch Modes', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_batch_mode_paired_comparison(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """Grouped bar chart: Real-Only vs each batch mode at each real data ratio."""
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    batch_modes = list(all_results.keys())
    first_bm = batch_modes[0]
    pairs = [
        ('exp11_real10', 'exp02_synth100_real10', '10%'),
        ('exp12_real25', 'exp03_synth100_real25', '25%'),
        ('exp13_real50', 'exp04_synth100_real50', '50%'),
        ('exp14_real75', 'exp05_synth100_real75', '75%'),
        ('exp07_real100', 'exp06_synth100_real100', '100%'),
    ]

    n_bars = 1 + len(batch_modes)
    bar_width = 0.8 / n_bars
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(pairs))

    # Real-only bars
    real_vals = []
    for real_exp, _, _ in pairs:
        m = all_results[first_bm].get(real_exp, {})
        real_vals.append(m.get('mAP', 0) if m.get('status') == 'completed' else 0)

    offset = -0.8 / 2 + bar_width / 2
    bars_real = ax.bar(x + offset, real_vals, bar_width, label='Real-Only',
                       color='#555555', edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars_real, real_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    bm_colors = plt.cm.Set2(np.linspace(0, 1, max(len(batch_modes), 3)))
    for i, bm in enumerate(batch_modes):
        synth_vals = []
        for _, synth_exp, _ in pairs:
            m = all_results[bm].get(synth_exp, {})
            synth_vals.append(m.get('mAP', 0) if m.get('status') == 'completed' else 0)
        bar_offset = offset + (i + 1) * bar_width
        bars_v = ax.bar(x + bar_offset, synth_vals, bar_width,
                        label=f'Synth+Real ({bm})', color=bm_colors[i],
                        edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars_v, synth_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('Real Data Percentage')
    ax.set_ylabel('mAP@0.5:0.95')
    ax.set_title('Paired Comparison: Real-Only vs Synth+Real by Batch Mode')
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_cross_batch_mode_report(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    synth_version: str,
    lang: str = 'en'
) -> None:
    """Generate a comprehensive markdown report comparing batch modes."""
    batch_modes = list(all_results.keys())
    is_en = lang == 'en'

    report = ""

    # Title
    report += f"# {'Batch Mode Comparison Report' if is_en else 'Reporte Comparativo de Modos de Batch'}\n\n"
    report += f"**{'Synthetic version' if is_en else 'Version sintetica'}**: {synth_version}\n\n"
    report += f"**{'Generated' if is_en else 'Generado'}**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**{'Batch modes compared' if is_en else 'Modos de batch comparados'}**: {', '.join(batch_modes)}\n\n"

    # --- Classify experiments ---
    mixed_experiments = [e for e in ALL_EXPERIMENTS if get_experiment_type(e) == 'mixed']
    invariant_experiments = [e for e in ALL_EXPERIMENTS if get_experiment_type(e) != 'mixed']

    # --- Executive Summary (only mixed experiments) ---
    report += f"## {'Executive Summary' if is_en else 'Resumen Ejecutivo'}\n\n"

    wins = {bm: 0 for bm in batch_modes}
    total_compared = 0
    all_deltas = []
    mixed_results = []  # Store for Key Insights section
    for exp in mixed_experiments:
        maps = {}
        for bm in batch_modes:
            m = all_results[bm].get(exp['name'], {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                maps[bm] = m['mAP']
        if len(maps) == len(batch_modes):
            best_bm = max(maps, key=maps.get)
            delta = max(maps.values()) - min(maps.values())
            wins[best_bm] += 1
            total_compared += 1
            all_deltas.append(delta)
            mixed_results.append({'exp': exp, 'maps': maps, 'best': best_bm, 'delta': delta})

    if total_compared > 0:
        if is_en:
            report += f"**Note**: Only {total_compared} mixed experiments (synth+real) are compared below. "
            report += f"{len(invariant_experiments)} experiments are batch-mode invariant "
            report += f"(synth-only or real-only) and produce identical results regardless of batch mode.\n\n"
        else:
            report += f"**Nota**: Solo se comparan {total_compared} experimentos mixtos (synth+real). "
            report += f"{len(invariant_experiments)} experimentos son invariantes al modo de batch "
            report += f"(solo sinteticos o solo reales) y producen resultados identicos en ambos modos.\n\n"
        for bm in batch_modes:
            pct = (wins[bm] / total_compared) * 100
            report += f"- **{bm}**: {wins[bm]}/{total_compared} "
            report += f"{'wins' if is_en else 'victorias'} ({pct:.0f}%)\n"
        avg_delta = sum(all_deltas) / len(all_deltas) if all_deltas else 0
        report += f"- **{'Average delta' if is_en else 'Delta promedio'}**: {avg_delta:.4f} mAP\n"
        report += f"- **{'Max delta' if is_en else 'Delta maximo'}**: {max(all_deltas):.4f} mAP\n"
    report += "\n"

    # --- Key Insights ---
    report += f"## {'Key Insights' if is_en else 'Hallazgos Clave'}\n\n"

    # Compute insights from mixed_results data
    # Separate by real data percentage pattern
    low_real_wins = []  # exp02, exp03, exp04 (10-50% real)
    high_real_wins = []  # exp05, exp06 (75-100% real)
    for mr in mixed_results:
        exp = mr['exp']
        real_r = exp.get('real_ratio', 0)
        real_r_val = float(real_r) if not isinstance(real_r, str) else 1.0
        if real_r_val <= 0.5:
            low_real_wins.append(mr)
        else:
            high_real_wins.append(mr)

    balanced_low_wins = sum(1 for mr in low_real_wins if mr['best'] == 'balanced')
    standard_high_wins = sum(1 for mr in high_real_wins if mr['best'] == 'standard')
    low_avg_delta = sum(mr['delta'] for mr in low_real_wins) / len(low_real_wins) if low_real_wins else 0
    high_avg_delta = sum(mr['delta'] for mr in high_real_wins) / len(high_real_wins) if high_real_wins else 0

    if is_en:
        report += f"1. **Balanced helps with scarce real data**: With 10-50% real data, balanced won "
        report += f"{balanced_low_wins}/{len(low_real_wins)} experiments (avg delta: {low_avg_delta:+.4f} mAP). "
        report += f"The DomainBalancedSampler ensures each batch contains real examples, preventing them "
        report += f"from being diluted by synthetic data.\n"
        report += f"2. **Standard wins with abundant real data**: With 75-100% real data, standard won "
        report += f"{standard_high_wins}/{len(high_real_wins)} experiments (avg delta: {high_avg_delta:+.4f} mAP). "
        report += f"The 50/50 constraint limits exposure to real data when it is plentiful.\n"
        report += f"3. **Crossover point**: The transition from balanced advantage to standard advantage "
        report += f"occurs around 50-75% real data.\n"

        # Training efficiency insight
        balanced_epochs = []
        standard_epochs = []
        for exp in mixed_experiments:
            for bm_name, epochs_list in [('balanced', balanced_epochs), ('standard', standard_epochs)]:
                m = all_results.get(bm_name, {}).get(exp['name'], {})
                t = m.get('training', {}) if isinstance(m.get('training'), dict) else {}
                ep = t.get('total_epochs')
                if ep is not None and isinstance(ep, (int, float)):
                    epochs_list.append(ep)
        if balanced_epochs and standard_epochs:
            avg_bal = sum(balanced_epochs) / len(balanced_epochs)
            avg_std = sum(standard_epochs) / len(standard_epochs)
            report += f"4. **Faster convergence with balanced**: Balanced mode converges in ~{avg_bal:.0f} epochs "
            report += f"on average vs ~{avg_std:.0f} for standard, suggesting more stable gradients per batch.\n"

        report += f"5. **Small overall deltas**: All differences are < {max(all_deltas):.3f} mAP, "
        report += f"indicating that batch composition strategy is a secondary factor compared to data quantity.\n"
    else:
        report += f"1. **Balanced ayuda con escasez de datos reales**: Con 10-50% de datos reales, balanced gano "
        report += f"{balanced_low_wins}/{len(low_real_wins)} experimentos (delta promedio: {low_avg_delta:+.4f} mAP). "
        report += f"El DomainBalancedSampler asegura que cada batch contenga ejemplos reales, evitando que se "
        report += f"diluyan entre los datos sinteticos.\n"
        report += f"2. **Standard gana con datos reales abundantes**: Con 75-100% de datos reales, standard gano "
        report += f"{standard_high_wins}/{len(high_real_wins)} experimentos (delta promedio: {high_avg_delta:+.4f} mAP). "
        report += f"La restriccion 50/50 limita la exposicion a datos reales cuando estos son abundantes.\n"
        report += f"3. **Punto de cruce**: La transicion de ventaja balanced a ventaja standard "
        report += f"ocurre alrededor de 50-75% de datos reales.\n"

        balanced_epochs = []
        standard_epochs = []
        for exp in mixed_experiments:
            for bm_name, epochs_list in [('balanced', balanced_epochs), ('standard', standard_epochs)]:
                m = all_results.get(bm_name, {}).get(exp['name'], {})
                t = m.get('training', {}) if isinstance(m.get('training'), dict) else {}
                ep = t.get('total_epochs')
                if ep is not None and isinstance(ep, (int, float)):
                    epochs_list.append(ep)
        if balanced_epochs and standard_epochs:
            avg_bal = sum(balanced_epochs) / len(balanced_epochs)
            avg_std = sum(standard_epochs) / len(standard_epochs)
            report += f"4. **Convergencia mas rapida con balanced**: Balanced converge en ~{avg_bal:.0f} epocas "
            report += f"en promedio vs ~{avg_std:.0f} para standard, sugiriendo gradientes mas estables por batch.\n"

        report += f"5. **Deltas pequenos**: Todas las diferencias son < {max(all_deltas):.3f} mAP, "
        report += f"indicando que la estrategia de composicion del batch es un factor secundario frente a la cantidad de datos.\n"
    report += "\n"

    # --- Methodology ---
    report += f"## {'Methodology' if is_en else 'Metodologia'}\n\n"
    if is_en:
        report += f"- **Standard batch mode**: Random sampling from the full training set\n"
        report += f"- **Balanced batch mode**: 50/50 real/synthetic sampling per batch using DomainBalancedSampler\n"
        report += f"- Both modes use the same datasets, model architecture, and hyperparameters\n"
        report += f"- The only variable is the batch composition strategy\n"
        report += f"- **Batch-mode invariant experiments** (synth-only or real-only) produce identical results "
        report += f"in both modes since there is only one domain to sample from. These are shown separately "
        report += f"for reference but excluded from the comparison analysis.\n"
    else:
        report += f"- **Modo standard**: Muestreo aleatorio del conjunto de entrenamiento completo\n"
        report += f"- **Modo balanced**: Muestreo 50/50 real/sintetico por batch usando DomainBalancedSampler\n"
        report += f"- Ambos modos usan los mismos datasets, arquitectura y hiperparametros\n"
        report += f"- La unica variable es la estrategia de composicion del batch\n"
        report += f"- **Experimentos invariantes al batch-mode** (solo sinteticos o solo reales) producen resultados "
        report += f"identicos en ambos modos ya que solo hay un dominio del que muestrear. Se muestran por separado "
        report += f"como referencia pero se excluyen del analisis comparativo.\n"
    report += "\n"

    # --- mAP Comparison Table: Mixed experiments (meaningful comparison) ---
    report += f"## {'mAP Comparison' if is_en else 'Comparacion mAP'}\n\n"
    report += f"### {'Mixed Experiments (meaningful comparison)' if is_en else 'Experimentos Mixtos (comparacion real)'}\n\n"
    if is_en:
        report += "These experiments use both synthetic and real data, so the DomainBalancedSampler produces different batch compositions.\n\n"
    else:
        report += "Estos experimentos usan datos sinteticos y reales, por lo que el DomainBalancedSampler produce composiciones de batch diferentes.\n\n"
    report += f"| {'Experiment' if is_en else 'Experimento'} | {'Strategy' if is_en else 'Estrategia'} |"
    for bm in batch_modes:
        report += f" mAP ({bm}) |"
    report += f" {'Best' if is_en else 'Mejor'} | Delta |\n"
    report += f"|{'---|' * (len(batch_modes) + 4)}\n"

    for exp in mixed_experiments:
        exp_name = exp['name']
        maps = {}
        for bm in batch_modes:
            m = all_results[bm].get(exp_name, {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                maps[bm] = m['mAP']
        if not maps:
            continue
        report += f"| {exp_name} | {exp['strategy']} |"
        for bm in batch_modes:
            val = maps.get(bm)
            report += f" {val:.4f} |" if val is not None else " - |"
        if len(maps) == len(batch_modes):
            best_bm = max(maps, key=maps.get)
            delta = maps[best_bm] - min(maps.values())
            report += f" **{best_bm}** | +{delta:.4f} |\n"
        else:
            report += " - | - |\n"

    # --- mAP Comparison Table: Invariant experiments (reference only) ---
    report += f"\n### {'Batch-Mode Invariant (reference only)' if is_en else 'Invariantes al Batch-Mode (solo referencia)'}\n\n"
    if is_en:
        report += "These experiments use only one data domain (synth-only or real-only). Results are identical across batch modes.\n\n"
    else:
        report += "Estos experimentos usan un solo dominio de datos (solo sinteticos o solo reales). Los resultados son identicos en ambos modos.\n\n"
    report += f"| {'Experiment' if is_en else 'Experimento'} | {'Type' if is_en else 'Tipo'} |"
    report += f" mAP |\n"
    report += f"|---|---|---|\n"

    for exp in invariant_experiments:
        exp_name = exp['name']
        # Get mAP from any batch mode (they're identical)
        m_val = None
        for bm in batch_modes:
            m = all_results[bm].get(exp_name, {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                m_val = m['mAP']
                break
        if m_val is None:
            continue
        exp_type = get_experiment_type(exp)
        type_label = exp_type.replace('_', '-')
        report += f"| {exp_name} | {type_label} | {m_val:.4f} |\n"

    # --- Visualizations references ---
    report += f"\n## {'Visualizations' if is_en else 'Visualizaciones'}\n\n"
    viz_files = [
        ('batch_mode_comparison.png', 'mAP Comparison' if is_en else 'Comparacion mAP'),
        ('batch_mode_ranking.png', 'Ranking Heatmap'),
        ('batch_mode_map_curves.png', 'mAP Curves' if is_en else 'Curvas mAP'),
        ('batch_mode_per_class_heatmap.png', 'Per-Class Heatmap' if is_en else 'Heatmap por Clase'),
        ('batch_mode_per_class_delta.png', 'Per-Class Delta' if is_en else 'Delta por Clase'),
        ('batch_mode_radar.png', 'Multi-Metric Radar'),
        ('batch_mode_minority_analysis.png', 'Minority Classes' if is_en else 'Clases Minoritarias'),
        ('batch_mode_training_efficiency.png', 'Training Efficiency' if is_en else 'Eficiencia de Entrenamiento'),
        ('batch_mode_domain_gap.png', 'Domain Gap' if is_en else 'Brecha de Dominio'),
        ('batch_mode_size_analysis.png', 'Size Analysis' if is_en else 'Analisis por Tamano'),
        ('batch_mode_synth_benefit.png', 'Synthetic Benefit' if is_en else 'Beneficio Sintetico'),
        ('batch_mode_paired_comparison.png', 'Paired Comparison' if is_en else 'Comparacion Pareada'),
    ]
    for fname, title in viz_files:
        report += f"### {title}\n\n![{title}]({fname})\n\n"

    # --- Per-Class Analysis (only mixed experiments) ---
    report += f"## {'Per-Class Analysis' if is_en else 'Analisis por Clase'}\n\n"
    if is_en:
        report += "Averaged over mixed experiments only (where batch mode produces different results).\n\n"
    else:
        report += "Promediado solo sobre experimentos mixtos (donde el modo de batch produce resultados diferentes).\n\n"
    mixed_exp_names = {e['name'] for e in mixed_experiments}
    report += f"| {'Class' if is_en else 'Clase'} |"
    for bm in batch_modes:
        report += f" {'Avg AP' if is_en else 'AP Prom'} ({bm}) |"
    report += f" {'Best Mode' if is_en else 'Mejor Modo'} | Delta |\n"
    report += f"|{'---|' * (len(batch_modes) + 3)}\n"

    for cls in CLASS_NAMES:
        report += f"| {cls} |"
        cls_avgs = {}
        for bm in batch_modes:
            vals = []
            for exp_name, m in all_results[bm].items():
                if exp_name not in mixed_exp_names:
                    continue
                if m.get('status') == 'completed':
                    ap = m.get(f'AP_{cls}')
                    if ap is not None:
                        vals.append(ap)
            avg = sum(vals) / len(vals) if vals else None
            cls_avgs[bm] = avg
            report += f" {avg:.4f} |" if avg is not None else " - |"
        valid = {k: v for k, v in cls_avgs.items() if v is not None}
        if len(valid) == len(batch_modes):
            best_bm = max(valid, key=valid.get)
            delta = max(valid.values()) - min(valid.values())
            report += f" **{best_bm}** | {delta:+.4f} |\n"
        else:
            report += " - | - |\n"

    # --- Minority Classes ---
    report += f"\n## {'Minority Class Analysis' if is_en else 'Analisis de Clases Minoritarias'}\n\n"
    if is_en:
        report += "Minority classes are those with fewer training samples. Balanced sampling is expected to help these classes most.\n\n"
    else:
        report += "Las clases minoritarias son aquellas con menos muestras de entrenamiento. El muestreo balanceado deberia beneficiar mas a estas clases.\n\n"

    for cls in MINORITY_CLASSES:
        report += f"### {cls}\n\n"
        report += f"| {'Experiment' if is_en else 'Experimento'} |"
        for bm in batch_modes:
            report += f" AP ({bm}) |"
        report += " Delta |\n"
        report += f"|{'---|' * (len(batch_modes) + 2)}\n"

        for exp in ALL_EXPERIMENTS:
            maps = {}
            for bm in batch_modes:
                m = all_results[bm].get(exp['name'], {})
                if m.get('status') == 'completed':
                    ap = m.get(f'AP_{cls}')
                    if ap is not None:
                        maps[bm] = ap
            if not maps:
                continue
            report += f"| {exp['name']} |"
            for bm in batch_modes:
                val = maps.get(bm)
                report += f" {val:.4f} |" if val is not None else " - |"
            if len(maps) == len(batch_modes):
                delta = max(maps.values()) - min(maps.values())
                report += f" {delta:+.4f} |\n"
            else:
                report += " - |\n"
        report += "\n"

    # --- Domain Gap ---
    report += f"## {'Domain Gap Analysis' if is_en else 'Analisis de Brecha de Dominio'}\n\n"
    for bm in batch_modes:
        sm = all_results[bm].get('exp01_synth100', {})
        rm = all_results[bm].get('exp07_real100', {})
        cm = all_results[bm].get('exp06_synth100_real100', {})
        s = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
        r = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
        c = cm.get('mAP', 0) if cm.get('status') == 'completed' else 0
        gap = r - s if (r > 0 and s > 0) else 0
        recovery = (c / r * 100) if r > 0 and c > 0 else 0
        report += f"- **{bm}**: "
        report += f"Synth-only={s:.4f}, Real-only={r:.4f}, "
        report += f"Gap={gap:.4f}, Combined={c:.4f}, "
        report += f"Recovery={recovery:.1f}%\n"
    report += "\n"

    # --- Synth Benefit ---
    report += f"## {'Synthetic Data Benefit' if is_en else 'Beneficio de Datos Sinteticos'}\n\n"
    if is_en:
        report += "Measures the mAP improvement from adding synthetic data compared to using real data alone. "
        report += "Real-only baselines are batch-mode invariant (identical across modes), so the delta reflects "
        report += "only the effect of the mixed training under each batch mode.\n\n"
    else:
        report += "Mide la mejora en mAP al agregar datos sinteticos comparado con usar solo datos reales. "
        report += "Las baselines de solo datos reales son invariantes al batch-mode (identicas en ambos modos), "
        report += "por lo que el delta refleja solo el efecto del entrenamiento mixto bajo cada modo de batch.\n\n"
    pairs = [
        ('exp02_synth100_real10', 'exp11_real10', '10%'),
        ('exp03_synth100_real25', 'exp12_real25', '25%'),
        ('exp04_synth100_real50', 'exp13_real50', '50%'),
        ('exp05_synth100_real75', 'exp14_real75', '75%'),
        ('exp06_synth100_real100', 'exp07_real100', '100%'),
    ]
    report += f"| {'Real %' if is_en else '% Real'} |"
    for bm in batch_modes:
        report += f" Delta ({bm}) | {'Rel %' if is_en else '% Rel'} ({bm}) |"
    report += "\n"
    report += f"|{'---|' * (len(batch_modes) * 2 + 1)}\n"

    for synth_exp, real_exp, label in pairs:
        report += f"| {label} |"
        for bm in batch_modes:
            sm = all_results[bm].get(synth_exp, {})
            rm = all_results[bm].get(real_exp, {})
            s = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            delta = s - r if (s > 0 and r > 0) else 0
            pct = ((s - r) / r * 100) if r > 0 and s > 0 else 0
            report += f" {delta:+.4f} | {pct:+.1f}% |"
        report += "\n"

    # --- Training Efficiency ---
    report += f"\n## {'Training Efficiency' if is_en else 'Eficiencia de Entrenamiento'}\n\n"
    report += f"| {'Experiment' if is_en else 'Experimento'} |"
    for bm in batch_modes:
        report += f" {'Duration' if is_en else 'Duracion'} ({bm}) | {'Epochs' if is_en else 'Epocas'} ({bm}) |"
    report += "\n"
    report += f"|{'---|' * (len(batch_modes) * 2 + 1)}\n"

    for exp in ALL_EXPERIMENTS:
        has_any = False
        row = f"| {exp['name']} |"
        for bm in batch_modes:
            m = all_results[bm].get(exp['name'], {})
            t = m.get('training', {}) if isinstance(m.get('training'), dict) else {}
            dur = t.get('duration_formatted', '-')
            epochs = t.get('total_epochs', '-')
            row += f" {dur} | {epochs} |"
            if dur != '-':
                has_any = True
        if has_any:
            report += row + "\n"

    # --- Conclusions ---
    report += f"\n## {'Conclusions' if is_en else 'Conclusiones'}\n\n"
    if total_compared > 0:
        avg_delta = sum(all_deltas) / len(all_deltas)

        # Compute per-regime stats for conclusions
        bal_low = sum(1 for mr in low_real_wins if mr['best'] == 'balanced')
        std_high = sum(1 for mr in high_real_wins if mr['best'] == 'standard')

        if is_en:
            report += f"Based on {total_compared} mixed experiments (excluding {len(invariant_experiments)} batch-mode invariant):\n\n"
            report += f"1. **Balanced sampling is beneficial with limited real data** (10-50%): "
            report += f"balanced won {bal_low}/{len(low_real_wins)} experiments in this regime. "
            report += f"By ensuring each batch contains real examples, the model learns the target domain more effectively.\n"
            report += f"2. **Standard sampling is preferable with abundant real data** (75-100%): "
            report += f"standard won {std_high}/{len(high_real_wins)} experiments. "
            report += f"The 50/50 constraint becomes counterproductive when real data is already well-represented.\n"
            report += f"3. **Performance differences are small**: average delta of {avg_delta:.4f} mAP "
            report += f"(max {max(all_deltas):.4f}), indicating that batch composition is a secondary factor "
            report += f"compared to overall data quantity and quality.\n"
            report += f"4. **Practical recommendation**: Use **balanced** mode when real data is scarce "
            report += f"(<50% of total training data). Use **standard** mode otherwise.\n"
        else:
            report += f"Basado en {total_compared} experimentos mixtos (excluyendo {len(invariant_experiments)} invariantes al batch-mode):\n\n"
            report += f"1. **El muestreo balanced es beneficioso con datos reales limitados** (10-50%): "
            report += f"balanced gano {bal_low}/{len(low_real_wins)} experimentos en este regimen. "
            report += f"Al asegurar que cada batch contenga ejemplos reales, el modelo aprende el dominio objetivo mas efectivamente.\n"
            report += f"2. **El muestreo standard es preferible con datos reales abundantes** (75-100%): "
            report += f"standard gano {std_high}/{len(high_real_wins)} experimentos. "
            report += f"La restriccion 50/50 se vuelve contraproducente cuando los datos reales ya estan bien representados.\n"
            report += f"3. **Las diferencias de rendimiento son pequenas**: delta promedio de {avg_delta:.4f} mAP "
            report += f"(max {max(all_deltas):.4f}), indicando que la composicion del batch es un factor secundario "
            report += f"comparado con la cantidad y calidad total de datos.\n"
            report += f"4. **Recomendacion practica**: Usar modo **balanced** cuando los datos reales son escasos "
            report += f"(<50% del total de datos de entrenamiento). Usar modo **standard** en caso contrario.\n"

    # --- Files Reference ---
    report += f"\n## {'Generated Files' if is_en else 'Archivos Generados'}\n\n"
    report += "```\n"
    for fname, title in viz_files:
        report += f"  {fname} - {title}\n"
    report += f"  batch_mode_comparison.csv - {'Data table' if is_en else 'Tabla de datos'}\n"
    report += f"  BATCH_MODE_REPORT.md - {'English report' if is_en else 'Reporte en ingles'}\n"
    report += f"  BATCH_MODE_REPORT_ES.md - {'Spanish report' if is_en else 'Reporte en espanol'}\n"
    report += "```\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {output_path}")


def run_cross_batch_mode_comparison(
    synth_version: str,
    output_dir: Path,
    no_plots: bool = False
) -> None:
    """
    Run complete cross-batch-mode comparison analysis.

    Args:
        synth_version: Synthetic dataset version to analyze
        output_dir: Output directory for comparison results
        no_plots: If True, skip generating plots
    """
    print("\n" + "="*70)
    print("CROSS-BATCH-MODE COMPARISON")
    print(f"Synthetic version: {synth_version}")
    print("="*70)

    print("\nCollecting results from all batch modes...")
    all_results = collect_all_batch_mode_results(synth_version)

    if not all_results:
        print("\nNo batch mode results found. Run experiments first.")
        return

    if len(all_results) < 2:
        print(f"\nOnly found results for {list(all_results.keys())}. Need at least 2 batch modes to compare.")
        return

    print(f"\nFound results for {len(all_results)} batch mode(s): {', '.join(all_results.keys())}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate CSV comparison
    print("\nGenerating comparison CSV...")
    csv_lines = ["experiment,strategy"]
    batch_modes = list(all_results.keys())
    for bm in batch_modes:
        csv_lines[0] += f",mAP_{bm}"
    csv_lines[0] += ",best_mode,delta"

    for exp in ALL_EXPERIMENTS:
        exp_name = exp['name']
        maps = {}
        for bm in batch_modes:
            m = all_results[bm].get(exp_name, {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                maps[bm] = m['mAP']

        if not maps:
            continue

        line = f"{exp_name},{exp['strategy']}"
        for bm in batch_modes:
            val = maps.get(bm)
            line += f",{val:.4f}" if val is not None else ","
        if len(maps) == len(batch_modes):
            best_bm = max(maps, key=maps.get)
            delta = maps[best_bm] - min(maps.values())
            line += f",{best_bm},{delta:.4f}"
        else:
            line += ",,"
        csv_lines.append(line)

    csv_path = output_dir / "batch_mode_comparison.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines))
    print(f"  Saved: {csv_path}")

    # Generate visualizations
    if not no_plots and HAS_MATPLOTLIB:
        print("\nGenerating cross-batch-mode visualizations...")
        plot_cross_batch_mode_comparison(all_results, output_dir / "batch_mode_comparison.png")
        plot_batch_mode_ranking(all_results, output_dir / "batch_mode_ranking.png")
        plot_batch_mode_map_curves(all_results, output_dir / "batch_mode_map_curves.png")
        plot_cross_batch_mode_per_class(all_results, output_dir / "batch_mode_per_class_heatmap.png")
        plot_batch_mode_per_class_delta(all_results, output_dir / "batch_mode_per_class_delta.png")
        plot_batch_mode_radar(all_results, output_dir / "batch_mode_radar.png")
        plot_batch_mode_minority_analysis(all_results, output_dir / "batch_mode_minority_analysis.png")
        plot_batch_mode_training_efficiency(all_results, output_dir / "batch_mode_training_efficiency.png")
        plot_batch_mode_domain_gap(all_results, output_dir / "batch_mode_domain_gap.png")
        plot_batch_mode_size_analysis(all_results, output_dir / "batch_mode_size_analysis.png")
        plot_batch_mode_synth_benefit(all_results, output_dir / "batch_mode_synth_benefit.png")
        plot_batch_mode_paired_comparison(all_results, output_dir / "batch_mode_paired_comparison.png")

    # Generate reports
    print("\nGenerating cross-batch-mode reports...")
    print("  Generating English report...")
    generate_cross_batch_mode_report(all_results, output_dir / "BATCH_MODE_REPORT.md", synth_version, lang='en')
    print("  Generating Spanish report...")
    generate_cross_batch_mode_report(all_results, output_dir / "BATCH_MODE_REPORT_ES.md", synth_version, lang='es')

    print("\n" + "="*70)
    print("CROSS-BATCH-MODE COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


# =============================================================================
# UNIFIED COMPARISON (cross-version × cross-batch-mode)
# =============================================================================


def collect_all_unified_results() -> Dict[str, Dict[str, Dict]]:
    """
    Collect results from all (version, batch_mode) combinations.

    Returns:
        Dict mapping config label (e.g. "densea/balanced") to experiment metrics dict.
        Only includes configurations with at least 1 completed experiment.
    """
    all_results = {}

    for version in get_available_synth_versions():
        for bm in BATCH_MODES:
            versioned_paths = get_versioned_output_paths(version, bm)
            results_dir = Path(versioned_paths['results'])

            # Check for state file
            state_file = results_dir / f"experiment_state_{version}.json"
            state_file_alt = results_dir / f"experiment_state_{version}_{bm}.json"

            if not (state_file.exists() or state_file_alt.exists()):
                continue

            config_label = f"{version}/{bm}"
            print(f"\n  Loading results for: {config_label}")
            metrics = collect_all_metrics(version, bm)

            completed = sum(1 for m in metrics.values() if m.get('status') == 'completed')
            if completed > 0:
                all_results[config_label] = metrics
                print(f"    {completed} completed experiments")
            else:
                print(f"    No completed experiments")

    return all_results


def _unified_plot_comparison(all_results, output_path):
    """Grouped bar chart: mAP by experiment for each configuration."""
    import matplotlib.pyplot as plt
    import numpy as np
    configs = list(all_results.keys())
    mixed_exps = [e for e in ALL_EXPERIMENTS if get_experiment_type(e) == 'mixed']

    fig, ax = plt.subplots(figsize=(max(14, len(mixed_exps) * 1.5), 7))
    x = np.arange(len(mixed_exps))
    width = 0.8 / len(configs)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(configs), 3)))

    for i, cfg in enumerate(configs):
        maps = []
        for exp in mixed_exps:
            m = all_results[cfg].get(exp['name'], {})
            maps.append(m.get('mAP', 0) if m.get('status') == 'completed' else 0)
        offset = (i - len(configs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, maps, width, label=cfg, color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, maps):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('mAP')
    ax.set_title('Unified Comparison: mAP by Experiment (mixed experiments only)')
    ax.set_xticks(x)
    ax.set_xticklabels([e['name'].replace('exp', 'E').split('_')[0] + '\n' + e['strategy']
                        for e in mixed_exps], fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.8, ax.get_ylim()[1] * 1.1))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _unified_plot_map_curves(all_results, output_path):
    """Line plot: mAP vs real data percentage for each configuration."""
    import matplotlib.pyplot as plt
    import numpy as np
    configs = list(all_results.keys())
    # Stratified experiments (exp02-exp06) paired with real-only baselines
    curve_exps = [
        ('exp02_synth100_real10', 0.1),
        ('exp03_synth100_real25', 0.25),
        ('exp04_synth100_real50', 0.5),
        ('exp05_synth100_real75', 0.75),
        ('exp06_synth100_real100', 1.0),
    ]
    real_only_exps = [
        ('exp11_real10', 0.1),
        ('exp12_real25', 0.25),
        ('exp13_real50', 0.5),
        ('exp14_real75', 0.75),
        ('exp07_real100', 1.0),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(configs), 3)))

    for i, cfg in enumerate(configs):
        ratios = []
        maps = []
        for exp_name, ratio in curve_exps:
            m = all_results[cfg].get(exp_name, {})
            if m.get('status') == 'completed' and m.get('mAP', 0) > 0:
                ratios.append(ratio * 100)
                maps.append(m['mAP'])
        if ratios:
            ax.plot(ratios, maps, 'o-', label=cfg, color=colors[i], linewidth=2, markersize=6)

    # Real-only baseline (same across configs, pick first)
    first_cfg = configs[0]
    ro_ratios, ro_maps = [], []
    for exp_name, ratio in real_only_exps:
        m = all_results[first_cfg].get(exp_name, {})
        if m.get('status') == 'completed' and m.get('mAP', 0) > 0:
            ro_ratios.append(ratio * 100)
            ro_maps.append(m['mAP'])
    if ro_ratios:
        ax.plot(ro_ratios, ro_maps, 's--', label='Real-only baseline', color='gray',
                linewidth=1.5, markersize=5, alpha=0.7)

    ax.set_xlabel('Real Data (%)')
    ax.set_ylabel('mAP')
    ax.set_title('mAP vs Real Data Ratio by Configuration')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks([10, 25, 50, 75, 100])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _unified_plot_per_class_heatmap(all_results, output_path):
    """Heatmap: per-class AP for best experiment across configurations."""
    import matplotlib.pyplot as plt
    import numpy as np
    configs = list(all_results.keys())
    best_exp = 'exp06_synth100_real100'

    data = []
    for cfg in configs:
        row = []
        m = all_results[cfg].get(best_exp, {})
        for cls in CLASS_NAMES:
            row.append(m.get(f'AP_{cls}', 0) if m.get('status') == 'completed' else 0)
        data.append(row)

    data_arr = np.array(data)
    fig, ax = plt.subplots(figsize=(max(10, len(CLASS_NAMES) * 1.2), max(4, len(configs) * 0.8)))
    im = ax.imshow(data_arr, cmap='YlOrRd', aspect='auto', vmin=0)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_title(f'Per-Class AP ({best_exp})')

    for i in range(len(configs)):
        for j in range(len(CLASS_NAMES)):
            val = data_arr[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _unified_plot_radar(all_results, output_path):
    """Radar chart: multi-metric comparison for best experiment."""
    import matplotlib.pyplot as plt
    import numpy as np
    configs = list(all_results.keys())
    best_exp = 'exp06_synth100_real100'
    metric_keys = ['mAP', 'mAP_50', 'mAP_75', 'AP_small', 'AP_medium', 'AP_large']
    metric_labels = ['mAP', 'mAP@50', 'mAP@75', 'AP small', 'AP medium', 'AP large']

    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(configs), 3)))

    for i, cfg in enumerate(configs):
        m = all_results[cfg].get(best_exp, {})
        values = [m.get(mk, 0) if m.get('status') == 'completed' else 0 for mk in metric_keys]
        values += values[:1]
        ax.plot(angles, values, 'o-', label=cfg, color=colors[i], linewidth=2)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title(f'Multi-Metric Radar ({best_exp})', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _unified_plot_domain_gap(all_results, output_path):
    """Domain gap analysis: synth-only vs real-only vs combined per config."""
    import matplotlib.pyplot as plt
    import numpy as np
    configs = list(all_results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(configs), 3)))
    x = np.arange(len(configs))

    synth_maps, real_maps, combined_maps = [], [], []
    for cfg in configs:
        s = all_results[cfg].get('exp01_synth100', {})
        r = all_results[cfg].get('exp07_real100', {})
        c = all_results[cfg].get('exp06_synth100_real100', {})
        synth_maps.append(s.get('mAP', 0) if s.get('status') == 'completed' else 0)
        real_maps.append(r.get('mAP', 0) if r.get('status') == 'completed' else 0)
        combined_maps.append(c.get('mAP', 0) if c.get('status') == 'completed' else 0)

    width = 0.25
    ax1.bar(x - width, synth_maps, width, label='Synth-only', color='#ff9999')
    ax1.bar(x, real_maps, width, label='Real-only', color='#99ccff')
    ax1.bar(x + width, combined_maps, width, label='Combined', color='#99ff99')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8, rotation=15)
    ax1.set_ylabel('mAP')
    ax1.set_title('Domain Gap Overview')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Recovery ratio
    recoveries = [(c / r * 100) if r > 0 else 0 for c, r in zip(combined_maps, real_maps)]
    ax2.bar(x, recoveries, color=[colors[i] for i in range(len(configs))], edgecolor='white')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% recovery')
    for i, (rec, cfg) in enumerate(zip(recoveries, configs)):
        ax2.text(i, rec + 0.5, f'{rec:.1f}%', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8, rotation=15)
    ax2.set_ylabel('Recovery (%)')
    ax2.set_title('Domain Gap Recovery (Combined / Real-only)')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _unified_plot_synth_benefit(all_results, output_path):
    """Synthetic data benefit: delta mAP (synth+real vs real-only) per config."""
    import matplotlib.pyplot as plt
    import numpy as np
    configs = list(all_results.keys())
    pairs = [
        ('exp02_synth100_real10', 'exp11_real10', '10%'),
        ('exp03_synth100_real25', 'exp12_real25', '25%'),
        ('exp04_synth100_real50', 'exp13_real50', '50%'),
        ('exp05_synth100_real75', 'exp14_real75', '75%'),
        ('exp06_synth100_real100', 'exp07_real100', '100%'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(configs), 3)))
    labels = [p[2] for p in pairs]

    for i, cfg in enumerate(configs):
        deltas, pcts = [], []
        for synth_exp, real_exp, _ in pairs:
            sm = all_results[cfg].get(synth_exp, {})
            rm = all_results[cfg].get(real_exp, {})
            s = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            d = s - r if (s > 0 and r > 0) else 0
            p = (d / r * 100) if r > 0 and s > 0 else 0
            deltas.append(d)
            pcts.append(p)
        ax1.plot(labels, deltas, 'o-', label=cfg, color=colors[i], linewidth=2)
        ax2.plot(labels, pcts, 'o-', label=cfg, color=colors[i], linewidth=2)

    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Real Data %')
    ax1.set_ylabel('Delta mAP')
    ax1.set_title('Synthetic Data Benefit (absolute)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Real Data %')
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.set_title('Synthetic Data Benefit (relative)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _unified_plot_ranking(all_results, output_path):
    """Ranking heatmap: mAP per experiment x configuration."""
    import matplotlib.pyplot as plt
    import numpy as np
    configs = list(all_results.keys())
    mixed_exps = [e for e in ALL_EXPERIMENTS if get_experiment_type(e) == 'mixed']

    data = []
    for cfg in configs:
        row = []
        for exp in mixed_exps:
            m = all_results[cfg].get(exp['name'], {})
            row.append(m.get('mAP', 0) if m.get('status') == 'completed' else 0)
        data.append(row)

    data_arr = np.array(data)
    fig, ax = plt.subplots(figsize=(max(10, len(mixed_exps) * 1.2), max(4, len(configs) * 0.8)))
    im = ax.imshow(data_arr, cmap='YlOrRd', aspect='auto')

    exp_labels = [e['name'].replace('exp0', 'E').replace('exp', 'E').split('_')[0]
                  for e in mixed_exps]
    ax.set_xticks(range(len(mixed_exps)))
    ax.set_xticklabels(exp_labels, fontsize=8)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_title('mAP Ranking (mixed experiments)')

    # Mark best per column
    for j in range(len(mixed_exps)):
        col = data_arr[:, j]
        if np.max(col) > 0:
            best_i = np.argmax(col)
            for i in range(len(configs)):
                val = data_arr[i, j]
                marker = ' *' if i == best_i else ''
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}{marker}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_unified_report(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    lang: str = 'en'
) -> None:
    """Generate a comprehensive unified comparison report (version x batch_mode)."""
    configs = list(all_results.keys())
    is_en = lang == 'en'

    mixed_experiments = [e for e in ALL_EXPERIMENTS if get_experiment_type(e) == 'mixed']
    invariant_experiments = [e for e in ALL_EXPERIMENTS if get_experiment_type(e) != 'mixed']
    mixed_exp_names = {e['name'] for e in mixed_experiments}

    report = ""

    # --- Title ---
    report += f"# {'Unified Comparison Report' if is_en else 'Reporte Comparativo Unificado'}\n\n"
    report += f"**{'Generated' if is_en else 'Generado'}**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**{'Configurations compared' if is_en else 'Configuraciones comparadas'}**: {', '.join(configs)}\n\n"

    # --- Executive Summary (only mixed experiments) ---
    report += f"## {'Executive Summary' if is_en else 'Resumen Ejecutivo'}\n\n"

    wins = {cfg: 0 for cfg in configs}
    total_compared = 0
    all_deltas = []
    mixed_results_data = []
    for exp in mixed_experiments:
        maps = {}
        for cfg in configs:
            m = all_results[cfg].get(exp['name'], {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                maps[cfg] = m['mAP']
        if len(maps) == len(configs):
            best_cfg = max(maps, key=maps.get)
            delta = max(maps.values()) - min(maps.values())
            wins[best_cfg] += 1
            total_compared += 1
            all_deltas.append(delta)
            mixed_results_data.append({'exp': exp, 'maps': maps, 'best': best_cfg, 'delta': delta})

    if total_compared > 0:
        if is_en:
            report += f"**Note**: Only {total_compared} mixed experiments (synth+real) are compared. "
            report += f"{len(invariant_experiments)} batch-mode invariant experiments are shown separately.\n\n"
        else:
            report += f"**Nota**: Solo se comparan {total_compared} experimentos mixtos (synth+real). "
            report += f"{len(invariant_experiments)} experimentos invariantes se muestran por separado.\n\n"
        for cfg in configs:
            pct = (wins[cfg] / total_compared) * 100
            report += f"- **{cfg}**: {wins[cfg]}/{total_compared} "
            report += f"{'wins' if is_en else 'victorias'} ({pct:.0f}%)\n"
        avg_delta = sum(all_deltas) / len(all_deltas)
        report += f"- **{'Average delta' if is_en else 'Delta promedio'}**: {avg_delta:.4f} mAP\n"
        report += f"- **{'Max delta' if is_en else 'Delta maximo'}**: {max(all_deltas):.4f} mAP\n"
    report += "\n"

    # --- Key Insights ---
    report += f"## {'Key Insights' if is_en else 'Hallazgos Clave'}\n\n"

    # Find overall best config
    if total_compared > 0:
        best_overall = max(wins, key=wins.get)
        avg_maps = {}
        for cfg in configs:
            cfg_maps = [mr['maps'][cfg] for mr in mixed_results_data]
            avg_maps[cfg] = sum(cfg_maps) / len(cfg_maps) if cfg_maps else 0
        best_avg_cfg = max(avg_maps, key=avg_maps.get)

        if is_en:
            report += f"1. **Best configuration by wins**: **{best_overall}** ({wins[best_overall]}/{total_compared} experiments)\n"
            report += f"2. **Best configuration by average mAP**: **{best_avg_cfg}** ({avg_maps[best_avg_cfg]:.4f} mAP)\n"

            # Version comparison insight
            versions_seen = set()
            for cfg in configs:
                v = cfg.split('/')[0]
                versions_seen.add(v)
            if len(versions_seen) > 1:
                report += f"3. **Synthetic data source matters more than batch mode**: "
                # Compare max delta across versions vs across batch modes
                version_deltas = []
                bm_deltas = []
                for mr in mixed_results_data:
                    maps = mr['maps']
                    by_version = {}
                    by_bm = {}
                    for cfg, val in maps.items():
                        v, bm = cfg.split('/')
                        by_version.setdefault(v, []).append(val)
                        by_bm.setdefault(bm, []).append(val)
                    for v, vals in by_version.items():
                        version_deltas.extend(vals)
                    for bm_name, vals in by_bm.items():
                        bm_deltas.extend(vals)
                # Compute cross-version spread vs cross-bm spread per experiment
                v_spreads = []
                bm_spreads = []
                for mr in mixed_results_data:
                    maps = mr['maps']
                    by_v = {}
                    by_b = {}
                    for cfg, val in maps.items():
                        v, bm = cfg.split('/')
                        by_v.setdefault(v, []).append(val)
                        by_b.setdefault(bm, []).append(val)
                    # avg per version, then spread
                    v_avgs = [sum(vals)/len(vals) for vals in by_v.values()]
                    b_avgs = [sum(vals)/len(vals) for vals in by_b.values()]
                    if len(v_avgs) > 1:
                        v_spreads.append(max(v_avgs) - min(v_avgs))
                    if len(b_avgs) > 1:
                        bm_spreads.append(max(b_avgs) - min(b_avgs))
                avg_v_spread = sum(v_spreads) / len(v_spreads) if v_spreads else 0
                avg_bm_spread = sum(bm_spreads) / len(bm_spreads) if bm_spreads else 0
                if avg_v_spread > avg_bm_spread:
                    report += f"Version choice has ~{avg_v_spread:.4f} mAP impact vs ~{avg_bm_spread:.4f} for batch mode.\n"
                else:
                    report += f"Batch mode has ~{avg_bm_spread:.4f} mAP impact vs ~{avg_v_spread:.4f} for version choice.\n"
        else:
            report += f"1. **Mejor configuracion por victorias**: **{best_overall}** ({wins[best_overall]}/{total_compared} experimentos)\n"
            report += f"2. **Mejor configuracion por mAP promedio**: **{best_avg_cfg}** ({avg_maps[best_avg_cfg]:.4f} mAP)\n"

            versions_seen = set()
            for cfg in configs:
                v = cfg.split('/')[0]
                versions_seen.add(v)
            if len(versions_seen) > 1:
                report += f"3. **La fuente de datos sinteticos importa mas que el modo de batch**: "
                v_spreads = []
                bm_spreads = []
                for mr in mixed_results_data:
                    maps = mr['maps']
                    by_v = {}
                    by_b = {}
                    for cfg, val in maps.items():
                        v, bm = cfg.split('/')
                        by_v.setdefault(v, []).append(val)
                        by_b.setdefault(bm, []).append(val)
                    v_avgs = [sum(vals)/len(vals) for vals in by_v.values()]
                    b_avgs = [sum(vals)/len(vals) for vals in by_b.values()]
                    if len(v_avgs) > 1:
                        v_spreads.append(max(v_avgs) - min(v_avgs))
                    if len(b_avgs) > 1:
                        bm_spreads.append(max(b_avgs) - min(b_avgs))
                avg_v_spread = sum(v_spreads) / len(v_spreads) if v_spreads else 0
                avg_bm_spread = sum(bm_spreads) / len(bm_spreads) if bm_spreads else 0
                if avg_v_spread > avg_bm_spread:
                    report += f"La version tiene ~{avg_v_spread:.4f} mAP de impacto vs ~{avg_bm_spread:.4f} para el modo de batch.\n"
                else:
                    report += f"El modo de batch tiene ~{avg_bm_spread:.4f} mAP de impacto vs ~{avg_v_spread:.4f} para la version.\n"
    report += "\n"

    # --- Methodology ---
    report += f"## {'Methodology' if is_en else 'Metodologia'}\n\n"
    if is_en:
        report += "Each configuration combines a synthetic data source with a batch sampling strategy:\n\n"
    else:
        report += "Cada configuracion combina una fuente de datos sinteticos con una estrategia de muestreo por batch:\n\n"
    report += f"| {'Configuration' if is_en else 'Configuracion'} | {'Synthetic Source' if is_en else 'Fuente Sintetica'} | {'Batch Mode' if is_en else 'Modo Batch'} | {'Classes' if is_en else 'Clases'} |\n"
    report += "|---|---|---|---|\n"
    for cfg in configs:
        parts = cfg.split('/')
        version, bm = parts[0], parts[1]
        try:
            synth_config = get_synth_config(version)
            name = synth_config['name']
            n_classes = synth_config.get('num_classes', '?')
        except Exception:
            name = version
            n_classes = '?'
        report += f"| {cfg} | {name} | {bm} | {n_classes} |\n"
    report += "\n"

    # --- mAP Comparison: Mixed experiments ---
    report += f"## {'mAP Comparison' if is_en else 'Comparacion mAP'}\n\n"
    report += f"### {'Mixed Experiments' if is_en else 'Experimentos Mixtos'}\n\n"
    report += f"| {'Experiment' if is_en else 'Experimento'} | {'Strategy' if is_en else 'Estrategia'} |"
    for cfg in configs:
        report += f" {cfg} |"
    report += f" {'Best' if is_en else 'Mejor'} | Delta |\n"
    report += f"|{'---|' * (len(configs) + 4)}\n"

    for exp in mixed_experiments:
        exp_name = exp['name']
        maps = {}
        for cfg in configs:
            m = all_results[cfg].get(exp_name, {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                maps[cfg] = m['mAP']
        if not maps:
            continue
        report += f"| {exp_name} | {exp['strategy']} |"
        for cfg in configs:
            val = maps.get(cfg)
            report += f" {val:.4f} |" if val is not None else " - |"
        if len(maps) == len(configs):
            best_cfg = max(maps, key=maps.get)
            delta = max(maps.values()) - min(maps.values())
            report += f" **{best_cfg}** | +{delta:.4f} |\n"
        else:
            report += " - | - |\n"

    # --- mAP: Invariant experiments ---
    report += f"\n### {'Batch-Mode Invariant (reference)' if is_en else 'Invariantes al Batch-Mode (referencia)'}\n\n"
    report += f"| {'Experiment' if is_en else 'Experimento'} | {'Type' if is_en else 'Tipo'} |"
    # Show one column per version (not per config, since batch mode doesn't matter)
    versions_in_configs = []
    for cfg in configs:
        v = cfg.split('/')[0]
        if v not in versions_in_configs:
            versions_in_configs.append(v)
    for v in versions_in_configs:
        report += f" {v} |"
    report += "\n"
    report += f"|{'---|' * (len(versions_in_configs) + 2)}\n"

    for exp in invariant_experiments:
        exp_name = exp['name']
        exp_type = get_experiment_type(exp).replace('_', '-')
        has_any = False
        row = f"| {exp_name} | {exp_type} |"
        for v in versions_in_configs:
            # Pick any config with this version
            val = None
            for cfg in configs:
                if cfg.startswith(v + '/'):
                    m = all_results[cfg].get(exp_name, {})
                    if m.get('status') == 'completed' and m.get('mAP') is not None:
                        val = m['mAP']
                        break
            row += f" {val:.4f} |" if val is not None else " - |"
            if val is not None:
                has_any = True
        if has_any:
            report += row + "\n"

    # --- Visualizations ---
    report += f"\n## {'Visualizations' if is_en else 'Visualizaciones'}\n\n"
    viz_files = [
        ('unified_comparison.png', 'mAP Comparison' if is_en else 'Comparacion mAP'),
        ('unified_map_curves.png', 'mAP Curves' if is_en else 'Curvas mAP'),
        ('unified_per_class_heatmap.png', 'Per-Class Heatmap' if is_en else 'Heatmap por Clase'),
        ('unified_radar.png', 'Multi-Metric Radar'),
        ('unified_domain_gap.png', 'Domain Gap' if is_en else 'Brecha de Dominio'),
        ('unified_synth_benefit.png', 'Synthetic Benefit' if is_en else 'Beneficio Sintetico'),
        ('unified_ranking.png', 'Ranking Heatmap'),
    ]
    for fname, title in viz_files:
        report += f"### {title}\n\n![{title}]({fname})\n\n"

    # --- Per-Class Analysis (mixed only) ---
    report += f"## {'Per-Class Analysis' if is_en else 'Analisis por Clase'}\n\n"
    if is_en:
        report += "Averaged over mixed experiments only.\n\n"
    else:
        report += "Promediado solo sobre experimentos mixtos.\n\n"
    report += f"| {'Class' if is_en else 'Clase'} |"
    for cfg in configs:
        report += f" {cfg} |"
    report += f" {'Best' if is_en else 'Mejor'} | Delta |\n"
    report += f"|{'---|' * (len(configs) + 3)}\n"

    for cls in CLASS_NAMES:
        report += f"| {cls} |"
        cls_avgs = {}
        for cfg in configs:
            vals = []
            for exp_name, m in all_results[cfg].items():
                if exp_name not in mixed_exp_names:
                    continue
                if m.get('status') == 'completed':
                    ap = m.get(f'AP_{cls}')
                    if ap is not None:
                        vals.append(ap)
            avg = sum(vals) / len(vals) if vals else None
            cls_avgs[cfg] = avg
            report += f" {avg:.4f} |" if avg is not None else " - |"
        valid = {k: v for k, v in cls_avgs.items() if v is not None}
        if len(valid) == len(configs):
            best_cfg = max(valid, key=valid.get)
            delta = max(valid.values()) - min(valid.values())
            report += f" **{best_cfg}** | {delta:+.4f} |\n"
        else:
            report += " - | - |\n"

    # --- Domain Gap ---
    report += f"\n## {'Domain Gap Analysis' if is_en else 'Analisis de Brecha de Dominio'}\n\n"
    for cfg in configs:
        sm = all_results[cfg].get('exp01_synth100', {})
        rm = all_results[cfg].get('exp07_real100', {})
        cm = all_results[cfg].get('exp06_synth100_real100', {})
        s = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
        r = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
        c = cm.get('mAP', 0) if cm.get('status') == 'completed' else 0
        gap = r - s if (r > 0 and s > 0) else 0
        recovery = (c / r * 100) if r > 0 and c > 0 else 0
        report += f"- **{cfg}**: Synth-only={s:.4f}, Real-only={r:.4f}, "
        report += f"Gap={gap:.4f}, Combined={c:.4f}, Recovery={recovery:.1f}%\n"
    report += "\n"

    # --- Synthetic Benefit ---
    report += f"## {'Synthetic Data Benefit' if is_en else 'Beneficio de Datos Sinteticos'}\n\n"
    pairs = [
        ('exp02_synth100_real10', 'exp11_real10', '10%'),
        ('exp03_synth100_real25', 'exp12_real25', '25%'),
        ('exp04_synth100_real50', 'exp13_real50', '50%'),
        ('exp05_synth100_real75', 'exp14_real75', '75%'),
        ('exp06_synth100_real100', 'exp07_real100', '100%'),
    ]
    report += f"| {'Real %' if is_en else '% Real'} |"
    for cfg in configs:
        report += f" Delta ({cfg}) | {'Rel %' if is_en else '% Rel'} |"
    report += "\n"
    report += f"|{'---|' * (len(configs) * 2 + 1)}\n"

    for synth_exp, real_exp, label in pairs:
        report += f"| {label} |"
        for cfg in configs:
            sm = all_results[cfg].get(synth_exp, {})
            rm = all_results[cfg].get(real_exp, {})
            s = sm.get('mAP', 0) if sm.get('status') == 'completed' else 0
            r = rm.get('mAP', 0) if rm.get('status') == 'completed' else 0
            delta = s - r if (s > 0 and r > 0) else 0
            pct = ((s - r) / r * 100) if r > 0 and s > 0 else 0
            report += f" {delta:+.4f} | {pct:+.1f}% |"
        report += "\n"

    # --- Conclusions ---
    report += f"\n## {'Conclusions' if is_en else 'Conclusiones'}\n\n"
    if total_compared > 0:
        avg_delta = sum(all_deltas) / len(all_deltas)
        best_overall = max(wins, key=wins.get)
        if is_en:
            report += f"Based on {total_compared} mixed experiments across {len(configs)} configurations:\n\n"
            report += f"1. **Best overall configuration**: **{best_overall}** "
            report += f"({wins[best_overall]}/{total_compared} wins, avg mAP: {avg_maps[best_overall]:.4f})\n"
            for cfg in configs:
                if cfg != best_overall:
                    report += f"2. **{cfg}**: {wins[cfg]}/{total_compared} wins, avg mAP: {avg_maps[cfg]:.4f}\n"
            report += f"3. **Performance spread**: avg delta {avg_delta:.4f} mAP, max {max(all_deltas):.4f} mAP\n"
        else:
            report += f"Basado en {total_compared} experimentos mixtos a traves de {len(configs)} configuraciones:\n\n"
            report += f"1. **Mejor configuracion global**: **{best_overall}** "
            report += f"({wins[best_overall]}/{total_compared} victorias, mAP promedio: {avg_maps[best_overall]:.4f})\n"
            for cfg in configs:
                if cfg != best_overall:
                    report += f"2. **{cfg}**: {wins[cfg]}/{total_compared} victorias, mAP promedio: {avg_maps[cfg]:.4f}\n"
            report += f"3. **Rango de rendimiento**: delta promedio {avg_delta:.4f} mAP, max {max(all_deltas):.4f} mAP\n"

    # --- Files Reference ---
    report += f"\n## {'Generated Files' if is_en else 'Archivos Generados'}\n\n"
    report += "```\n"
    for fname, title in viz_files:
        report += f"  {fname} - {title}\n"
    report += f"  unified_comparison.csv - {'Data table' if is_en else 'Tabla de datos'}\n"
    report += f"  UNIFIED_REPORT.md - {'English report' if is_en else 'Reporte en ingles'}\n"
    report += f"  UNIFIED_REPORT_ES.md - {'Spanish report' if is_en else 'Reporte en espanol'}\n"
    report += "```\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {output_path}")


def run_unified_comparison(
    output_dir: Path,
    no_plots: bool = False
) -> None:
    """
    Run unified comparison across all versions and batch modes.

    Treats each (version, batch_mode) pair as a distinct configuration.
    """
    print("\n" + "="*70)
    print("UNIFIED COMPARISON (version x batch_mode)")
    print("="*70)

    print("\nCollecting results from all configurations...")
    all_results = collect_all_unified_results()

    if len(all_results) < 2:
        print(f"\nNeed at least 2 configurations to compare (found {len(all_results)}).")
        print("Run experiments with different versions or batch modes first.")
        return

    configs = list(all_results.keys())
    print(f"\nFound {len(configs)} configuration(s): {', '.join(configs)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate CSV
    print("\nGenerating comparison CSV...")
    csv_lines = ["experiment,strategy"]
    for cfg in configs:
        csv_lines[0] += f",mAP_{cfg}"
    csv_lines[0] += ",best_config,delta"

    for exp in ALL_EXPERIMENTS:
        maps = {}
        for cfg in configs:
            m = all_results[cfg].get(exp['name'], {})
            if m.get('status') == 'completed' and m.get('mAP') is not None:
                maps[cfg] = m['mAP']
        if not maps:
            continue
        row = f"{exp['name']},{exp['strategy']}"
        for cfg in configs:
            val = maps.get(cfg)
            row += f",{val:.4f}" if val is not None else ","
        if len(maps) == len(configs):
            best = max(maps, key=maps.get)
            delta = max(maps.values()) - min(maps.values())
            row += f",{best},{delta:.4f}"
        else:
            row += ",,"
        csv_lines.append(row)

    csv_path = output_dir / "unified_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"  Saved: {csv_path}")

    # Generate visualizations
    if not no_plots and HAS_MATPLOTLIB:
        print("\nGenerating unified visualizations...")
        _unified_plot_comparison(all_results, output_dir / "unified_comparison.png")
        _unified_plot_map_curves(all_results, output_dir / "unified_map_curves.png")
        _unified_plot_per_class_heatmap(all_results, output_dir / "unified_per_class_heatmap.png")
        _unified_plot_radar(all_results, output_dir / "unified_radar.png")
        _unified_plot_domain_gap(all_results, output_dir / "unified_domain_gap.png")
        _unified_plot_synth_benefit(all_results, output_dir / "unified_synth_benefit.png")
        _unified_plot_ranking(all_results, output_dir / "unified_ranking.png")

    # Generate reports
    print("\nGenerating unified reports...")
    print("  Generating English report...")
    generate_unified_report(all_results, output_dir / "UNIFIED_REPORT.md", lang='en')
    print("  Generating Spanish report...")
    generate_unified_report(all_results, output_dir / "UNIFIED_REPORT_ES.md", lang='es')

    print("\n" + "="*70)
    print("UNIFIED COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect results and generate visualizations"
    )
    parser.add_argument(
        '--synth-version', '-v',
        choices=get_available_synth_versions(),
        default=DEFAULT_SYNTH_VERSION,
        help=f'Synthetic dataset version to analyze (default: {DEFAULT_SYNTH_VERSION})'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: versioned results directory)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--compare-versions',
        action='store_true',
        help='Compare results across all synthetic dataset versions'
    )
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help='List available synthetic dataset versions and exit'
    )
    parser.add_argument(
        '--batch-mode',
        choices=BATCH_MODES,
        default=DEFAULT_BATCH_MODE,
        help=f'Batch sampling mode for results collection (default: {DEFAULT_BATCH_MODE})'
    )
    parser.add_argument(
        '--compare-batch-modes',
        action='store_true',
        help='Compare results across all batch modes for a given synthetic version'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare results across all versions and batch modes combined (unified comparison)'
    )

    args = parser.parse_args()

    # List versions and exit
    if args.list_versions:
        print("\nAvailable synthetic dataset versions:")
        print("-" * 70)
        for version, config in SYNTHETIC_DATASETS.items():
            default_marker = " (default)" if version == DEFAULT_SYNTH_VERSION else ""
            print(f"  {version}{default_marker}: {config['name']}")
            print(f"    {config['description']}")
        return

    batch_mode = args.batch_mode

    # Cross-batch-mode comparison
    if args.compare_batch_modes:
        output_dir = Path(args.output_dir) if args.output_dir else Path(
            get_versioned_output_paths(args.synth_version)['results']
        ).parent / "comparison_batch_modes"
        run_cross_batch_mode_comparison(args.synth_version, output_dir, args.no_plots)
        return

    # Unified comparison (all versions × all batch modes)
    if args.compare_all:
        output_dir = Path(args.output_dir) if args.output_dir else Path(RESULTS_OUTPUT_DIR).parent / "comparison_unified"
        run_unified_comparison(output_dir, args.no_plots)
        return

    # Cross-version comparison mode
    if args.compare_versions:
        output_dir = Path(args.output_dir) if args.output_dir else Path(RESULTS_OUTPUT_DIR)
        run_cross_version_comparison(output_dir, args.no_plots, batch_mode)
        return

    # Standard single-version analysis
    synth_version = args.synth_version
    versioned_paths = get_versioned_output_paths(synth_version, batch_mode)

    # Use versioned output directory by default
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(versioned_paths['results'])

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("COLLECTING EXPERIMENT RESULTS")
    print(f"Synthetic version: {synth_version}")
    print(f"Batch mode: {batch_mode}")
    print("="*70)

    # Collect metrics
    print("\nExtracting metrics from evaluation outputs...")
    metrics = collect_all_metrics(synth_version, batch_mode)

    # Create DataFrame
    df = create_pandas_dataframe(metrics)

    # Save CSV
    print("\nGenerating comparison table...")
    csv_path = output_dir / "comparison_table.csv"
    csv_content = create_comparison_table(metrics)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    print(f"  Saved: {csv_path}")

    # Generate visualizations
    if not args.no_plots and HAS_MATPLOTLIB and df is not None:
        print("\nGenerating visualizations...")

        plot_map_vs_real_ratio(df, output_dir / "mAP_vs_real_ratio.png")
        plot_minority_class_analysis(df, output_dir / "minority_class_analysis.png")
        plot_per_class_comparison(df, output_dir / "per_class_comparison.png")

        # Paired comparison visualizations (real-only vs synth+real)
        print("\nGenerating paired comparison visualizations...")
        plot_paired_comparison(df, output_dir / "paired_comparison.png")
        plot_synth_benefit_curve(df, output_dir / "synth_benefit_analysis.png")

        # Comprehensive training strategy comparisons
        print("\nGenerating comprehensive strategy comparison visualizations...")
        plot_training_strategy_comparison(df, output_dir / "training_strategy_comparison.png")
        plot_per_class_strategy_comparison(df, output_dir / "per_class_strategy_comparison.png")

    # Generate paired comparison CSV
    print("\nGenerating paired comparison table...")
    create_paired_comparison_csv(metrics, output_dir / "comparison_pairs.csv")

    # Generate reports in both languages
    print("\nGenerating markdown reports...")

    # English report
    print("  Generating English report...")
    generate_markdown_report(df, metrics, output_dir / "EXPERIMENT_REPORT.md", lang='en', synth_version=synth_version, batch_mode=batch_mode)

    # Spanish report
    print("  Generating Spanish report...")
    generate_markdown_report(df, metrics, output_dir / "EXPERIMENT_REPORT_ES.md", lang='es', synth_version=synth_version, batch_mode=batch_mode)

    print("\n" + "="*70)
    print("RESULTS COLLECTION COMPLETE")
    print(f"Synthetic version: {synth_version}")
    print(f"Batch mode: {batch_mode}")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")

    print("\nTo compare results across all versions, run:")
    print("  python experiments/collect_results.py --compare-versions")
    print("\nTo compare results across batch modes, run:")
    print(f"  python experiments/collect_results.py -v {synth_version} --compare-batch-modes")
    print("\nTo compare all versions × batch modes combined, run:")
    print("  python experiments/collect_results.py --compare-all")


if __name__ == '__main__':
    main()

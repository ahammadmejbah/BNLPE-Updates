``` python


├── tailwind.config.js
├── postcss.config.js
├── package.json

├── agentic_docai/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py

├── templates/
│   ├── base.html
│   ├── includes/
│   │   ├── navbar.html
│   │   └── footer.html
│   ├── interface/
│   │   ├── home.html
│   │   ├── modules.html
│   │   └── settings_panel.html
│   ├── print/
│   │   ├── summary_report.html
│   │   ├── layout_preview.html
│   │   └── key_extractions.html

├── static/
│   ├── css/tailwind.css
│   └── js/main.js

├── core/
│   ├── engine.py
│   ├── dispatcher.py
│   ├── orchestrator.py
│   ├── pipeline.py
│   ├── config.yaml
│   └── routing_logic.py

├── preprocessing/
│   ├── grayscale.py
│   ├── binarize.py
│   ├── deskew.py
│   ├── noise_removal.py
│   └── contrast_enhancement.py

├── segmentation/
│   ├── line_segmentation.py
│   ├── layout_analyzer.py
│   ├── table_segmentation.py
│   └── region_segmenter.py

├── recognition/
│   ├── symbol_matcher.py
│   ├── bangla_unicode_mapper.py
│   ├── language_model.py
│   └── recognition_postprocess.py

├── correction/
│   ├── error_classifier.py
│   ├── feedback_analyzer.py
│   └── contextual_corrector.py

├── rules/
│   ├── rule_0001_to_0050/
│   ├── rule_0051_to_0100/
│   ├── ...
│   └── rule_1451_to_1500/

├── ocr_math_rules/
│   ├── matrix_multiplication.py
│   ├── transformer_math_encoder.py
│   ├── sobel_edge_detection.py
│   └── ... (700+ symbolic rules)

├── datasets/
│   ├── training/
│   │   ├── images/
│   │   ├── audio/
│   │   ├── documents/
│   │   │   ├── invoices/
│   │   │   ├── transcripts/
│   │   │   ├── certificates/
│   │   │   └── contracts/
│   │   └── labels/
│   │       ├── ocr_labels.json
│   │       ├── segment_annotations.csv
│   │       ├── audio_transcripts.json
│   │       └── video_frame_mappings.json
│   └── evaluation/
│       ├── test_set_v1/
│       └── benchmarks/
│           ├── benchmark_paper_tasks.csv
│           └── multimodal_ocr_testcases.json

├── transformer/
│   ├── architecture/
│   │   ├── attention.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── activation.py
│   │   └── positional_encoding.py
│   ├── adapters/
│   │   ├── ridge_regression_adapter.py
│   │   └── rail_fusion.py
│   ├── prompt_learning/
│   │   ├── aggregator.py
│   │   ├── generator.py
│   │   └── projection_head.py
│   ├── training/
│   │   ├── dataset_loader.py
│   │   ├── trainer.py
│   │   └── scheduler.py
│   └── reward_modeling/
│       ├── prompt_rewards.py
│       ├── rmab_simulator.py
│       └── evaluator.py

├── llm/
│   ├── tokenizer.py
│   ├── model_config.py
│   ├── embedding_layer.py
│   ├── transformer_stack.py
│   ├── language_head.py
│   ├── loss_functions.py
│   ├── generation.py
│   ├── pipeline.py
│   ├── export_utils.py
│   ├── inference_interface.py
│   ├── vision_adapter.py
│   ├── multimodal_fuser.py
│   ├── continual_learning_adapter.py
│   ├── alignment_monitor.py
│   ├── hallucination_detector.py
│   ├── calibration_module.py
│   ├── eval_suite.py
│   ├── logging_utils.py
│   ├── checkpoint_manager.py
│   ├── retrieval_interface.py
│   ├── attention_visualizer.py
│   ├── tokenizer_analyzer.py
│   ├── contextual_bias_checker.py
│   ├── document_layout_encoder.py
│   ├── token_confidence_mapper.py
│   ├── scalability_profiler.py
│   ├── token_latency_tracker.py
│   ├── token_streaming_engine.py
│   ├── real_time_adapter.py
│   ├── fine_tuning_tools.py
│   ├── synthetic_data_generator.py
│   ├── gradient_flow_debugger.py
│   ├── autoregressive_scorer.py
│   ├── latent_concept_mapper.py
│   └── benchmark_tests/
│       ├── truthfulness_eval.py
│       ├── consistency_eval.py
│       └── multilingual_support_check.py

├── agent/
│   ├── think_loop.py
│   ├── plan_act_reflect.py
│   ├── self_debugger.py
│   ├── memory_engine.py
│   ├── emergent_communication.py
│   ├── curriculum_scheduler.py
│   ├── alignment_evaluator.py
│   ├── reflection_journal.py
│   ├── feedback_policy_adapter.py
│   ├── autonomous_planner.py
│   ├── error_feedback_loop.py
│   ├── self_optimization_engine.py
│   ├── goal_state_translator.py
│   ├── decision_trace_logger.py
│   ├── policy_reward_refiner.py
│   ├── anomaly_reasoning_module.py
│   └── collaborative_agent_bridge.py

├── graph_reasoning/
│   ├── algorithms/
│   │   ├── dfs.py
│   │   ├── bfs.py
│   │   └── reachability_encoder.py
│   ├── environments/
│   │   ├── blocksworld.json
│   │   └── theorem_graphs/
│   └── planner.py

├── imitation_learning/
│   ├── policy_network.py
│   ├── language_goal_mapper.py
│   └── multimodal_controller.py

├── compression/
│   ├── alpha_pruning/
│   │   ├── compute_esd.py
│   │   ├── pl_alpha_estimator.py
│   │   └── layerwise_sparsity_allocator.py
│   └── pruning_strategies/
│       ├── sparsegpt.py
│       ├── wanda.py
│       └── structured_pruning.py

├── apps/
│   ├── voice/
│   │   ├── recorder.py
│   │   └── audio_to_text.py
│   ├── documents/
│   │   └── doc_parser.py
│   ├── tables/
│   │   └── table_extractor.py
│   ├── summary/
│   │   └── summarizer.py
│   ├── export/
│   │   ├── export_to_pdf.py
│   │   └── export_to_csv.py
│   ├── retriever/
│   │   └── similarity_search.py
│   └── debug/
│       ├── model_probe.py
│       └── ocr_failure_cases.py

├── analytics/
│   ├── usage_logs/
│   ├── performance/
│   │   ├── confusion_matrix.py
│   │   └── fairness_metrics.py
│   └── visualizations/
│       ├── graph_attn_map.py
│       ├── seaborn_charts.py
│       └── token_entropy_plot.py





```

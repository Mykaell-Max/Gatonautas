import React from "react";

type ClassificationRow = {
	label: string;
	precision: string;
	recall: string;
	f1Score: string;
	support: string;
	highlighted?: boolean;
};

const keyMetrics = [
	{
		label: "Accuracy",
		value: "0.94",
		helper: "Overall proportion of correct predictions across 52 samples.",
	},
	{
		label: "Precision (weighted)",
		value: "0.95",
		helper: "Average precision weighted by class support.",
	},
	{
		label: "Recall (weighted)",
		value: "0.94",
		helper: "Average recall weighted by the number of examples per class.",
	},
	{
		label: "F1-score (weighted)",
		value: "0.94",
		helper: "Harmonic mean balancing precision and recall for imbalanced data.",
	},
];

const hyperparameters = [
	{ label: "n_estimators", value: "100" },
	{ label: "learning_rate", value: "0.1" },
	{ label: "max_depth", value: ">5" },
	{ label: "num_leaves", value: "31" },
	{ label: "min_child_samples", value: "20" },
	{ label: "random_state", value: "None" },
	{ label: "boosting_type", value: "gbdt" },
	{ label: "subsample", value: "0.8" },
	{ label: "colsample_bytree", value: "0.3" },
	{ label: "reg_alpha", value: "0" },
	{ label: "reg_lambda", value: "0.5" },
];

const classificationReport: ClassificationRow[] = [
	{ label: "NON CANDIDATE", precision: "0.98", recall: "0.95", f1Score: "0.97", support: "44" },
	{ label: "CANDIDATE", precision: "0.78", recall: "0.88", f1Score: "0.82", support: "8" },
	{ label: "accuracy", precision: "—", recall: "—", f1Score: "0.94", support: "52", highlighted: true },
	{ label: "macro avg", precision: "0.88", recall: "0.91", f1Score: "0.89", support: "52" },
	{ label: "weighted avg", precision: "0.95", recall: "0.94", f1Score: "0.94", support: "52" },
];

const ModelMetricsView: React.FC = () => {
	return (
		<div className="metrics-container">
			<header className="metrics-header">
				<span className="badge">LightGBM</span>
				<h1>LightGBM Model Metrics</h1>
				<p>
					These mock metrics summarise the latest LightGBM classification run on the
					curated exoplanet dataset. They capture predictive quality, class balance
					and the hyperparameter signature used for this configuration.
				</p>
			</header>

			<section className="metrics-grid">
				{keyMetrics.map((metric) => (
					<article className="metric-card" key={metric.label}>
						<h2>{metric.label}</h2>
						<span className="metric-value">{metric.value}</span>
						<p>{metric.helper}</p>
					</article>
				))}
				<article className="metric-card">
					<h2>Classification score signature</h2>
					<p>
						Sequence exported from the mock training pipeline describing the
						hyperparameter snapshot for this LightGBM run:
					</p>
					<dl className="hyperparameters-list">
						{hyperparameters.map((param) => (
							<React.Fragment key={param.label}>
								<dt>{param.label}</dt>
								<dd>{param.value}</dd>
							</React.Fragment>
						))}
					</dl>
				</article>
			</section>

			<section className="metrics-table-wrapper">
				<div className="metrics-table-title">
					<h2>Classification Report</h2>
					<span>Support = 52 samples</span>
				</div>
				<table className="classification-table">
					<thead>
						<tr>
							<th>Label</th>
							<th>Precision</th>
							<th>Recall</th>
							<th>F1-Score</th>
							<th>Support</th>
						</tr>
					</thead>
					<tbody>
						{classificationReport.map((row) => (
							<tr key={row.label} className={row.highlighted ? "highlight" : undefined}>
								<td>{row.label}</td>
								<td>{row.precision}</td>
								<td>{row.recall}</td>
								<td>{row.f1Score}</td>
								<td>{row.support}</td>
							</tr>
						))}
					</tbody>
				</table>
			</section>

			<footer className="metrics-footnote">
				These figures are provided as illustrative mock data for the LightGBM model.
				They allow the UI to be validated before wiring live backend metrics.
			</footer>
		</div>
	);
};

export default ModelMetricsView;

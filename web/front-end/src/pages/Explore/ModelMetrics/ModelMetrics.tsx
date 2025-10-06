import React from "react";
import ModelMetricsView from "./ModelMetricsView";
import "./ModelMetrics.css";

const ModelMetrics: React.FC = () => {
	return (
		<div className="metrics-page">
			<ModelMetricsView />
		</div>
	);
};

export default ModelMetrics;

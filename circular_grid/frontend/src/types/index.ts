// ============================================================
// Circular Grid - API TypeScript Interfaces
// ============================================================

// ── Campus Overview ──────────────────────────────────────

export type BuildingStatus = 'online' | 'stale' | 'offline';

export interface SparkPoint {
  ts: string;       // ISO timestamp
  value: number;
}

export interface BuildingSummary {
  id: string;
  name: string;
  status: BuildingStatus;
  electricity_kwh: number;
  electricity_estimated: boolean;
  water_litres: number;
  water_estimated: boolean;
  footfall: number;
  footfall_estimated: boolean;
  sparkline: SparkPoint[];  // last 24h electricity data points
  offline_since?: string;   // ISO timestamp, present when status === 'offline'
}

export interface CampusOverviewResponse {
  updated_at: string;
  buildings: BuildingSummary[];
}

// ── Building Detail / Forecast ────────────────────────────

export type ForecastStream = 'electricity_kwh' | 'water_litres' | 'footfall';
export type ForecastHorizon = '6h' | '12h';

export interface ForecastPoint {
  ts: string;
  actual: number | null;
  forecast: number;
  lower: number;
  upper: number;
}

export interface CalendarEvent {
  ts: string;
  label: string;
}

export interface BuildingForecastResponse {
  building_id: string;
  building_name: string;
  stream: ForecastStream;
  horizon: ForecastHorizon;
  series: ForecastPoint[];
  calendar_events: CalendarEvent[];
}

// ── Anomalies ─────────────────────────────────────────────

export interface Cause {
  name: string;
  contribution: number;   // 0..1
  evidence: string;
}

export interface Recommendation {
  action: string;
  savings_kwh?: number;
  savings_litres?: number;
  savings_kg?: number;
  savings_rs?: number;
  savings_co2_kg?: number;
  confidence: number;  // 0..1
  why: string;
}

export interface Anomaly {
  anomaly_id: string;
  building_id: string;
  building_name: string;
  stream: string;
  timestamp: string;
  z_score: number;
  causes: Cause[];
  recommendations: Recommendation[];
}

export interface AnomaliesResponse {
  anomalies: Anomaly[];
}

// ── Waste Allocation ──────────────────────────────────────

export interface BuildingWaste {
  building_id: string;
  building_name: string;
  total_kg: number;
  composted_kg: number;
  recycled_kg: number;
  landfill_kg: number;
  diversion_rate: number;  // 0..1
}

export interface WasteAllocationResponse {
  total_generated_kg: number;
  total_composted_kg: number;
  total_recycled_kg: number;
  total_landfill_kg: number;
  overall_diversion_rate: number;
  compost_ratio: string;
  landfill_avoided_kg: number;
  buildings: BuildingWaste[];
}

// ── Edge Classifier ───────────────────────────────────────

export interface ClassifierEvent {
  event_id: string;
  building_id: string;
  building_name: string;
  timestamp: string;
  predicted_class: string;
  confidence: number;   // 0..1
  latency_ms: number;
}

export interface ModelInfo {
  model_id: string;
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  deployed: boolean;
}

export interface ClassifierFeedResponse {
  events: ClassifierEvent[];
  models: ModelInfo[];
}

export interface ClassifyRequest {
  building_id: string;
  stream: string;
  value: number;
}

export interface ClassifyResponse {
  predicted_class: string;
  confidence: number;
  latency_ms: number;
}

// ── Connectivity ──────────────────────────────────────────

export interface ConnectivityBuilding {
  building_id: string;
  building_name: string;
  status: BuildingStatus;
  offline_since?: string;
  last_seen: string;
  catching_up: boolean;
  catchup_progress?: number;  // 0..1
}

export interface ConnectivityResponse {
  buildings: ConnectivityBuilding[];
}

// ── Data Quality ──────────────────────────────────────────

export interface DataQualityBuilding {
  building_id: string;
  building_name: string;
  duplicates_removed: number;
  gaps_detected: number;
  imputed_values: number;
  quality_score: number;  // 0..1
}

export interface DataQualityResponse {
  buildings: DataQualityBuilding[];
}

// ── Leaderboard ───────────────────────────────────────────

export interface CreditBreakdown {
  segregation: number;
  peak_reduction: number;
  landfill_avoided: number;
}

export interface LeaderboardEntry {
  rank: number;
  building_id: string;
  building_name: string;
  normalized_intensity: number;  // per occupant
  credit_score: number;
  credit_breakdown: CreditBreakdown;
  anti_gaming_flag: boolean;
}

export interface LeaderboardResponse {
  period: string;
  entries: LeaderboardEntry[];
}

// ── Cost & Scale ──────────────────────────────────────────

export interface HardwareLine {
  item: string;
  quantity: number;
  unit_cost_rs: number;
  total_cost_rs: number;
  assumed: boolean;
}

export interface ScaleTestResult {
  buildings: number;
  ingestion_throughput_msg_per_sec: number;
  query_latency_p50_ms: number;
  query_latency_p99_ms: number;
}

export interface CostScaleResponse {
  hardware_budget: HardwareLine[];
  total_hardware_cost_rs: number;
  scale_tests: ScaleTestResult[];
}

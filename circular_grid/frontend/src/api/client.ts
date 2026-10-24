// ============================================================
// Circular Grid - Typed API Client
// ============================================================

import type {
  CampusOverviewResponse,
  BuildingForecastResponse,
  ForecastStream,
  ForecastHorizon,
  AnomaliesResponse,
  WasteAllocationResponse,
  ClassifierFeedResponse,
  ClassifyRequest,
  ClassifyResponse,
  ConnectivityResponse,
  DataQualityResponse,
  LeaderboardResponse,
  CostScaleResponse,
} from '../types';

const BASE = '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`GET ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

async function post<TBody, TResponse>(path: string, body: TBody): Promise<TResponse> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`POST ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<TResponse>;
}

export const api = {
  getCampusOverview(): Promise<CampusOverviewResponse> {
    return get('/campus/overview');
  },

  getBuildingForecast(
    id: string,
    stream: ForecastStream = 'electricity_kwh',
    horizon: ForecastHorizon = '6h'
  ): Promise<BuildingForecastResponse> {
    return get(`/building/${id}/forecast?stream=${stream}&horizon=${horizon}`);
  },

  getAnomalies(): Promise<AnomaliesResponse> {
    return get('/anomalies');
  },

  getWasteAllocation(): Promise<WasteAllocationResponse> {
    return get('/waste/allocation');
  },

  getClassifierFeed(): Promise<ClassifierFeedResponse> {
    return get('/classifier/feed');
  },

  classify(body: ClassifyRequest): Promise<ClassifyResponse> {
    return post('/classifier/classify', body);
  },

  getConnectivity(): Promise<ConnectivityResponse> {
    return get('/connectivity');
  },

  getDataQuality(): Promise<DataQualityResponse> {
    return get('/data-quality');
  },

  getLeaderboard(): Promise<LeaderboardResponse> {
    return get('/leaderboard');
  },

  getCostScale(): Promise<CostScaleResponse> {
    return get('/cost-scale');
  },
};

import { useState } from 'react';
import CampusOverview from './pages/CampusOverview';
import BuildingDetail from './pages/BuildingDetail';
import { api } from './api/client';
import type { ClassifierFeedResponse } from './types';

type Page = 'overview' | 'classifier';

export default function App() {
  const [page, setPage] = useState<Page>('overview');
  const [buildingId, setBuildingId] = useState<string | null>(null);
  const [feed, setFeed] = useState<ClassifierFeedResponse | null>(null);
  const [classifierError, setClassifierError] = useState<string | null>(null);

  async function showClassifier() {
    setPage('classifier');
    setBuildingId(null);
    setClassifierError(null);
    try {
      setFeed(await api.getClassifierFeed());
    } catch (error) {
      setClassifierError(error instanceof Error ? error.message : 'Unable to load classifier feed');
    }
  }

  return (
    <div className="app-layout">
      <nav className="app-nav">
        <div className="app-nav__inner">
          <button className="app-nav__brand nav-link" onClick={() => { setPage('overview'); setBuildingId(null); }} type="button">
            Circular Grid
          </button>
          <div className="app-nav__links">
            <button className={`nav-link ${page === 'overview' && !buildingId ? 'nav-link--active' : ''}`} onClick={() => { setPage('overview'); setBuildingId(null); }} type="button">Campus</button>
            <button className={`nav-link ${page === 'classifier' ? 'nav-link--active' : ''}`} onClick={showClassifier} type="button">Waste classifier</button>
          </div>
        </div>
      </nav>
      <main className="app-main">
        {buildingId ? <BuildingDetail buildingId={buildingId} onBack={() => setBuildingId(null)} /> : page === 'overview' ? (
          <CampusOverview onSelectBuilding={(id) => setBuildingId(id)} />
        ) : (
          <section>
            <div className="page-header"><h1>Waste classifier</h1><p>Predictions from the waste-events test file using sensor features.</p></div>
            {classifierError && <div className="error-state">{classifierError}</div>}
            {feed && <div className="card">
              <div className="section-header"><h2>Recent test events</h2><span className="text-muted">{feed.events.length} events</span></div>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table"><thead><tr><th>Event</th><th>Building</th><th>Prediction</th><th>Confidence</th></tr></thead>
                  <tbody>{feed.events.map((event) => <tr key={event.event_id}><td>{event.event_id}</td><td>{event.building_name}</td><td>{event.predicted_class}</td><td>{(event.confidence * 100).toFixed(0)}%</td></tr>)}</tbody>
                </table>
              </div>
            </div>}
          </section>
        )}
      </main>
    </div>
  );
}

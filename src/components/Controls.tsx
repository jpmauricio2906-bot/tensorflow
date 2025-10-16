import React from 'react'
import type { Settings, DatasetKind, ActivationKind } from '../types'

export default function Controls(p:{ s:Settings; set:(patch:Partial<Settings>)=>void; onAddLayer:()=>void; onRemoveLayer:(idx:number)=>void; onChangeLayerUnits:(idx:number,units:number)=>void; onReset:()=>void; onTrainToggle:()=>void; onRegenData:()=>void; }){
  const { s } = p
  const dsOptions: DatasetKind[] = ['circle','xor','gaussian','spiral','linear']
  const acts: ActivationKind[] = ['tanh','relu','sigmoid','linear']

  return (
    <aside className="sidebar">
      <div className="section">
        <h3>Dataset</h3>
        <div className="row">
          <label>Type</label>
          <select value={s.dataset} onChange={e=>p.set({dataset: e.target.value as any})}>
            {dsOptions.map(d=><option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div className="row">
          <label>Samples</label>
          <input type="number" min={100} max={5000} value={s.samples} onChange={e=>p.set({samples: +e.target.value})}/>
        </div>
        <div className="row">
          <label>Noise</label>
          <input type="range" min={0} max={0.5} step={0.01} value={s.noise} onChange={e=>p.set({noise: +e.target.value})}/>
          <div className="small">{s.noise.toFixed(2)}</div>
        </div>
        <div className="row">
          <label>Train %</label>
          <input type="range" min={10} max={90} step={1} value={Math.round(s.trainSplit*100)} onChange={e=>p.set({trainSplit: (+e.target.value)/100})}/>
          <div className="small">{Math.round(s.trainSplit*100)}%</div>
        </div>
        <div className="row">
          <label>Show test data</label>
          <input type="checkbox" checked={s.showTest} onChange={e=>p.set({showTest: e.target.checked})}/>
        </div>
        <div className="btns">
          <button className="alt" onClick={p.onRegenData}>Regenerate Data</button>
        </div>
      </div>

      <div className="section">
        <h3>Features</h3>
        <div className="checkbox-grid">
          {[
            ['x','x'], ['y','y'], ['xy','x*y'], ['x2','x^2'], ['y2','y^2'],
            ['sinX','sin(xπ)'], ['cosX','cos(xπ)'], ['sinY','sin(yπ)'], ['cosY','cos(yπ)'],
          ].map(([k,label])=>(
            <label key={k} style={{display:'flex',gap:6,alignItems:'center'}}>
              <input type="checkbox" checked={(s.features as any)[k]} onChange={e=>p.set({features:{...s.features,[k]: e.target.checked}})}/>
              {label}
            </label>
          )) as any}
        </div>
        <div className="row">
          <label>Grid (px/cell)</label>
          <input type="range" min={3} max={16} step={1} value={s.gridResolution} onChange={e=>p.set({gridResolution: +e.target.value})}/>
          <div className="small">{s.gridResolution}</div>
        </div>
      </div>

      <div className="section">
        <h3>Network</h3>
        <div className="row">
          <label>Activation</label>
          <select value={s.activation} onChange={e=>p.set({activation: e.target.value as any})}>
            {acts.map(a=><option key={a} value={a}>{a}</option>)}
          </select>
        </div>
        <div className="layer-editor">
          {s.layers.map((u,idx)=>(
            <div key={idx} style={{display:'flex',gap:6,alignItems:'center',background:'#1a2043',padding:'6px 8px',borderRadius:8}}>
              <span className="small">L{idx+1}</span>
              <input type="number" min={1} max={64} value={u} onChange={e=>p.onChangeLayerUnits(idx, Math.max(1, Math.min(64, +e.target.value)) )}/>
              <button className="danger" onClick={()=>p.onRemoveLayer(idx)}>−</button>
            </div>
          ))}
          <button onClick={p.onAddLayer}>+ Add layer</button>
        </div>
      </div>

      <div className="section">
        <h3>Training</h3>
        <div className="row">
          <label>Learning rate</label>
          <input type="range" min={0.0005} max={0.3} step={0.0005} value={s.learningRate} onChange={e=>p.set({learningRate: +e.target.value})}/>
          <div className="small">{s.learningRate.toFixed(4)}</div>
        </div>
        <div className="row">
          <label>Batch size</label>
          <input type="number" min={4} max={512} value={s.batchSize} onChange={e=>p.set({batchSize: Math.max(4, Math.min(512, +e.target.value))})}/>
        </div>
        <div className="row">
          <label>L2 regularization</label>
          <input type="range" min={0} max={0.02} step={0.0005} value={s.l2} onChange={e=>p.set({l2: +e.target.value})}/>
          <div className="small">{s.l2.toFixed(4)}</div>
        </div>

        <div className="btns">
          <button className="primary" onClick={p.onTrainToggle}>{s.running ? 'Pause' : 'Train'}</button>
          <button onClick={p.onReset}>Reset Model</button>
        </div>
      </div>

      <div className="small">Tip: change the dataset & features, then press “Train”.</div>
    </aside>
  )
}

import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import Controls from './components/Controls'
import CanvasView from './components/CanvasView'
import type { Settings } from './types'
import { genCircle, genGaussian, genLinear, genSpiral, genXor } from './lib/datasets'
import { createModel, makeFeatureFn, featureDim, toTensors } from './lib/model'

const defaultSettings: Settings = {
  dataset: 'circle',
  samples: 800,
  noise: 0.05,
  trainSplit: 0.5,
  showTest: true,
  layers: [8, 4],
  activation: 'tanh',
  learningRate: 0.03,
  l2: 0,
  batchSize: 32,
  features: { x:true, y:true, xy:false, x2:false, y2:false, sinX:false, cosX:false, sinY:false, cosY:false },
  gridResolution: 8,
  running: false,
  problemType: 'classification', // NEW
}

function makeData(s: Settings){
  switch (s.dataset){
    case 'circle': return genCircle(s.samples, s.noise)
    case 'xor': return genXor(s.samples, s.noise)
    case 'gaussian': return genGaussian(s.samples, s.noise)
    case 'spiral': return genSpiral(s.samples, s.noise)
    default: return genLinear(s.samples, s.noise)
  }
}

// Fisher–Yates shuffle utility to ensure **random spatial distribution**
function shuffleInUnison(xs: number[][], ys: number[]) {
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
    [ys[i], ys[j]] = [ys[j], ys[i]];
  }
}

export default function App(){
  const [s, setS] = useState<Settings>(defaultSettings)
  const [data, setData] = useState<{xs:number[][], ys:number[]} | null>(null)
  const [splitIdx, setSplitIdx] = useState(0)
  const [status, setStatus] = useState('idle')
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const trainLoopRef = useRef<number | null>(null)
  const epochRef = useRef(0)

  const regenData = React.useCallback(()=>{
    const d = makeData(s)

    // NEW: randomize order so train/test are evenly distributed on the chart
    const xs = d.xs.slice()
    const ys = d.ys.slice()
    shuffleInUnison(xs, ys)

    const split = Math.floor(xs.length * s.trainSplit)
    setData({ xs, ys }); setSplitIdx(split)
  }, [s.dataset, s.samples, s.noise, s.trainSplit])

  const rebuildModel = React.useCallback(()=>{
    if (model) { model.dispose(); setModel(null) }
    const dim = featureDim(s.features)
    const m = createModel(s, dim)
    setModel(m); epochRef.current = 0
  }, [s.layers.join(','), s.activation, s.learningRate, s.l2, s.features, s.problemType])

  useEffect(()=>{ regenData() }, []) // initial
  useEffect(()=>{ rebuildModel() }, [])

  useEffect(()=>{ regenData() }, [s.dataset, s.samples, s.noise, s.trainSplit])
  useEffect(()=>{ rebuildModel() }, [s.layers, s.activation, s.learningRate, s.l2, s.features, s.problemType])

  useEffect(()=>{
    if (!s.running || !model || !data) return
    const m = model as tf.LayersModel
    const feat = makeFeatureFn(s.features)
    const trainXY = data.xs.slice(0, splitIdx)
    const trainY  = data.ys.slice(0, splitIdx)
    const { X, Y } = toTensors(trainXY, trainY, feat)
    let cancelled = false
    async function step(){
      if (cancelled) return
      const h = await m.fit(X, Y, { epochs: 1, batchSize: s.batchSize, shuffle: true, verbose: 0 })
      epochRef.current += 1
      const loss = (h.history.loss?.[0] as number)?.toFixed(4)
      // Show metric based on problem type
      const metricKey = s.problemType === 'classification' ? 'acc' : 'mse'
      const metricVal = (h.history[metricKey as keyof typeof h.history]?.[0] as number | undefined)
      const metricStr = metricVal !== undefined
        ? (s.problemType === 'classification' ? `${(metricVal*100).toFixed(1)}%` : metricVal.toFixed(4))
        : '—'
      setStatus(
        s.problemType === 'classification'
          ? `epoch ${epochRef.current}  |  loss ${loss}  |  acc ${metricStr}`
          : `epoch ${epochRef.current}  |  loss ${loss}  |  mse ${metricStr}`
      )
      if (s.running) trainLoopRef.current = requestAnimationFrame(step)
    }
    trainLoopRef.current = requestAnimationFrame(step)
    return ()=>{ cancelled = true; if (trainLoopRef.current) cancelAnimationFrame(trainLoopRef.current); X.dispose(); Y.dispose() }
  }, [s.running, model, data, splitIdx, s.batchSize, s.features, s.problemType])

  const set = (patch: Partial<Settings>) => setS(prev => ({...prev, ...patch}))
  const onAddLayer = () => setS(prev=>({...prev, layers: [...prev.layers, 4]}))
  const onRemoveLayer = (idx:number) => setS(prev=>{ const next = prev.layers.slice(); next.splice(idx,1); return {...prev, layers: next} })
  const onChangeLayerUnits = (idx:number, units:number) => setS(prev=>{ const next = prev.layers.slice(); next[idx] = units; return {...prev, layers: next} })
  const onReset = ()=>{ setS(prev=>({...prev, running:false})); rebuildModel() }
  const onTrainToggle = ()=> setS(prev=>({...prev, running: !prev.running}))
  const onRegenData = ()=> regenData()

  useEffect(()=>{ if (!data) return; const split = Math.floor(data.xs.length * s.trainSplit); setSplitIdx(split) }, [data, s.trainSplit])

  return (
    <div className="app">
      <Controls
        s={s} set={set}
        onAddLayer={onAddLayer}
        onRemoveLayer={onRemoveLayer}
        onChangeLayerUnits={onChangeLayerUnits}
        onReset={onReset}
        onTrainToggle={onTrainToggle}
        onRegenData={onRegenData}
      />
      <CanvasView settings={s} model={model} data={data} splitIndex={splitIdx} status={status} />
    </div>
  )
}

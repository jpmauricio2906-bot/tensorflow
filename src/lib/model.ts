import * as tf from '@tensorflow/tfjs'
import type { FeaturesToggle, Settings } from '../types'

export function makeFeatureFn(t: FeaturesToggle){
  return (x: number, y: number) => {
    const f: number[] = []
    if (t.x) f.push(x)
    if (t.y) f.push(y)
    if (t.xy) f.push(x*y)
    if (t.x2) f.push(x*x)
    if (t.y2) f.push(y*y)
    if (t.sinX) f.push(Math.sin(x*Math.PI))
    if (t.cosX) f.push(Math.cos(x*Math.PI))
    if (t.sinY) f.push(Math.sin(y*Math.PI))
    if (t.cosY) f.push(Math.cos(y*Math.PI))
    return f.length===0 ? [x,y] : f
  }
}

export function featureDim(t: FeaturesToggle){
  const vals = [t.x,t.y,t.xy,t.x2,t.y2,t.sinX,t.cosX,t.sinY,t.cosY]
  return Math.max(vals.reduce((a,b)=>a+(b?1:0),0), 2)
}

export function createModel(s: Settings, inputDim: number){
  const m = tf.sequential()
  const reg = s.l2 > 0 ? tf.regularizers.l2({ l2: s.l2 }) : undefined

  if (s.layers.length === 0){
    m.add(tf.layers.dense({ inputShape:[inputDim], units:1, activation:'sigmoid', kernelRegularizer: reg, biasRegularizer: reg }))
  } else {
    m.add(tf.layers.dense({ inputShape:[inputDim], units:s.layers[0], activation: s.activation, kernelRegularizer: reg, biasRegularizer: reg }))
    for (let i=1;i<s.layers.length;i++){
      m.add(tf.layers.dense({ units:s.layers[i], activation: s.activation, kernelRegularizer: reg, biasRegularizer: reg }))
    }
    m.add(tf.layers.dense({ units:1, activation:'sigmoid', kernelRegularizer: reg, biasRegularizer: reg }))
  }

  m.compile({ optimizer: tf.train.adam(s.learningRate), loss:'binaryCrossentropy', metrics:['accuracy'] })
  return m
}

export function toTensors(xy: number[][], y: number[], feat: (x:number,y:number)=>number[]){
  const X = tf.tensor2d(xy.map(([x1,y1])=>feat(x1,y1)), undefined, 'float32')
  const Y = tf.tensor2d(y.map(v=>[v]))
  return { X, Y }
}

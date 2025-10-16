import * as tf from '@tensorflow/tfjs'
import type { FeaturesToggle, Settings } from '../types'

export function makeFeatureFn(toggles: FeaturesToggle) {
  return (x: number, y: number): number[] => {
    const f: number[] = [];
    if (toggles.x) f.push(x);
    if (toggles.y) f.push(y);
    if (toggles.xy) f.push(x*y);
    if (toggles.x2) f.push(x*x);
    if (toggles.y2) f.push(y*y);
    if (toggles.sinX) f.push(Math.sin(x*Math.PI));
    if (toggles.cosX) f.push(Math.cos(x*Math.PI));
    if (toggles.sinY) f.push(Math.sin(y*Math.PI));
    if (toggles.cosY) f.push(Math.cos(y*Math.PI));
    if (f.length===0) return [x,y];
    return f;
  };
}

export function featureDim(t: FeaturesToggle): number {
  const vals = [t.x, t.y, t.xy, t.x2, t.y2, t.sinX, t.cosX, t.sinY, t.cosY];
  const count = vals.reduce((a,b)=>a+(b?1:0),0);
  return Math.max(count, 2);
}

export function createModel(settings: Settings, inputDim: number) {
  const { layers, activation, learningRate, l2 } = settings;
  const model = tf.sequential();
  const reg = l2 > 0 ? tf.regularizers.l2({ l2 }) : undefined;

  if (layers.length === 0) {
    model.add(tf.layers.dense({
      inputShape: [inputDim],
      units: 1, activation: 'sigmoid',
      kernelRegularizer: reg, biasRegularizer: reg,
    }));
  } else {
    model.add(tf.layers.dense({
      inputShape: [inputDim],
      units: layers[0], activation,
      kernelRegularizer: reg, biasRegularizer: reg,
    }));
    for (let i=1;i<layers.length;i++){
      model.add(tf.layers.dense({
        units: layers[i], activation,
        kernelRegularizer: reg, biasRegularizer: reg,
      }));
    }
    model.add(tf.layers.dense({
      units: 1, activation: 'sigmoid',
      kernelRegularizer: reg, biasRegularizer: reg,
    }));
  }

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

export function toTensors(
  xy: number[][], y: number[], feat: (x:number,y:number)=>number[]
){
  const sample = feat(0,0);
  const X = tf.tensor2d(xy.map(([x1,y1]) => feat(x1,y1)), undefined, 'float32');
  const Y = tf.tensor2d(y.map(v => [v]));
  return { X, Y };
}

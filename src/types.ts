export type DatasetKind = 'circle' | 'xor' | 'gaussian' | 'spiral' | 'linear';
export type ActivationKind = 'tanh' | 'relu' | 'sigmoid' | 'linear';

export interface FeaturesToggle {
  x: boolean; y: boolean; x2: boolean; y2: boolean; xy: boolean;
  sinX: boolean; cosX: boolean; sinY: boolean; cosY: boolean;
}

export interface Settings {
  dataset: DatasetKind;
  samples: number;
  noise: number;
  trainSplit: number;
  showTest: boolean;
  layers: number[];
  activation: ActivationKind;
  learningRate: number;
  l2: number;
  batchSize: number;
  features: FeaturesToggle;
  gridResolution: number;
  running: boolean;
}

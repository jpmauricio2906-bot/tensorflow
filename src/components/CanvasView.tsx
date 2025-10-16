import React, { useEffect, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'
import { clear, drawDecision, drawPoints } from '../lib/draw'
import type { Settings } from '../types'

export default function CanvasView({ settings, model, data, splitIndex, status }:{ settings:Settings; model:tf.LayersModel|null; data:{xs:number[][], ys:number[]} | null; splitIndex:number; status:string; }){
  const canvasRef = useRef<HTMLCanvasElement|null>(null)

  useEffect(()=>{
    const canvas = canvasRef.current
    if (!canvas || !data) return
    const ctx = canvas.getContext('2d')!
    const w = canvas.clientWidth, h = canvas.clientHeight
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.floor(w*dpr); canvas.height = Math.floor(h*dpr)
    ctx.scale(dpr, dpr)
    clear(ctx, w, h)
    ;(async ()=>{
      await drawDecision({ ctx, width:w, height:h, model, settings })
      drawPoints(ctx, w, h, data, splitIndex, settings.showTest)
    })()
  }, [settings, model, data, splitIndex])

  return (
    <div className="canvas-wrap">
      <canvas ref={canvasRef} className="canvas"/>
      <div className="legend">
        <span><span className="dot dotA"></span> Class 1</span>
        <span><span className="dot dotB"></span> Class 0</span>
        <span className="status">{status}</span>
      </div>
    </div>
  )
}

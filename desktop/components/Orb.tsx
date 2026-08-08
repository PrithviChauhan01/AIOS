"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export type OrbState = "idle" | "listening" | "speaking";

const ACTIVITY: Record<OrbState, number> = { idle: 0, listening: 0.45, speaking: 1.0 };

const VERT = `varying vec3 vN;varying vec3 vView;void main(){vN=normal;vec4 mv=modelViewMatrix*vec4(position,1.0);vView=normalize(-mv.xyz);gl_Position=projectionMatrix*mv;}`;

const FRAG = `precision highp float;varying vec3 vN;varying vec3 vView;
uniform float uTime,uActivity,uBeat,uFront;uniform vec3 uColorA,uColorB,uHot;
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x,289.0);}vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){const vec2 C=vec2(1.0/6.0,1.0/3.0);const vec4 D=vec4(0.0,0.5,1.0,2.0);vec3 i=floor(v+dot(v,C.yyy));vec3 x0=v-i+dot(i,C.xxx);vec3 g=step(x0.yzx,x0.xyz);vec3 l=1.0-g;vec3 i1=min(g.xyz,l.zxy);vec3 i2=max(g.xyz,l.zxy);vec3 x1=x0-i1+C.xxx;vec3 x2=x0-i2+C.yyy;vec3 x3=x0-D.yyy;i=mod(i,289.0);vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));float n_=1.0/7.0;vec3 ns=n_*D.wyz-D.xzx;vec4 j=p-49.0*floor(p*ns.z*ns.z);vec4 x_=floor(j*ns.z);vec4 y_=floor(j-7.0*x_);vec4 x=x_*ns.x+ns.yyyy;vec4 y=y_*ns.x+ns.yyyy;vec4 h=1.0-abs(x)-abs(y);vec4 b0=vec4(x.xy,y.xy);vec4 b1=vec4(x.zw,y.zw);vec4 s0=floor(b0)*2.0+1.0;vec4 s1=floor(b1)*2.0+1.0;vec4 sh=-step(h,vec4(0.0));vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;vec3 p0=vec3(a0.xy,h.x);vec3 p1=vec3(a0.zw,h.y);vec3 p2=vec3(a1.xy,h.z);vec3 p3=vec3(a1.zw,h.w);vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);m=m*m;return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));}
float fbm(vec3 p){float s=0.0,a=0.55;for(int i=0;i<5;i++){s+=a*snoise(p);p*=2.02;a*=0.5;}return s;}
void main(){
  float t=uTime*(0.18+uActivity*0.6);vec3 n=normalize(vN);
  vec3 q=n*(1.7+uActivity*0.7);
  float warp=fbm(q+vec3(t*0.4,t*0.3,-t*0.35));
  float f=fbm(q*1.35+warp*1.8+vec3(-t*0.5,t*0.4,t*0.25));
  float ridge=1.0-abs(f);
  ridge=pow(ridge,9.0+uActivity*3.0);
  ridge=smoothstep(0.5,0.78,ridge);
  float ridge2=1.0-abs(warp);
  ridge2=pow(ridge2,8.0);
  ridge2=smoothstep(0.55,0.82,ridge2);
  float veins=clamp(ridge*1.2+ridge2*0.7,0.0,1.6);
  float fres=pow(1.0-max(dot(n,vView),0.0),2.0);
  float mixv=0.5+0.5*sin(f*2.6+t+n.y*2.0);
  vec3 plasma=mix(uColorA,uColorB,mixv);
  vec3 lightA=mix(uColorA,vec3(1.0),0.55);
  vec3 lightB=mix(uColorB,vec3(1.0),0.55);
  vec3 lightTint=mix(lightA,lightB,mixv);
  vec3 col=plasma*veins;
  col+=lightTint*pow(veins,4.0)*(0.35+uActivity*0.4);
  col+=plasma*fres*0.28;
  col*=0.72+uBeat*0.35+uActivity*0.2;
  col=pow(col,vec3(1.3));
  float depthDim=mix(0.28,1.0,uFront);col*=depthDim;col+=plasma*fres*uFront*0.4;
  float alpha=clamp(veins*0.85+fres*0.75,0.0,1.0);alpha*=mix(0.55,1.0,uFront);
  if(alpha<0.01)discard;gl_FragColor=vec4(col,alpha);
}`;

export default function Orb({ state = "idle" }: { state?: OrbState }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const targetRef = useRef(ACTIVITY[state]);

  useEffect(() => {
    targetRef.current = ACTIVITY[state];
  }, [state]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    cam.position.z = 3.4;

    // Camera distance is chosen so the sphere fills ~65% of whichever
    // dimension is the tighter constraint (width for a narrow rail, height
    // for a wide/short one) — roughly 17-18% margin per side, so the sphere
    // reads as dominant without touching the rail's bounds.
    function fitOrb() {
      const r = canvas!.getBoundingClientRect();
      const aspect = r.width / Math.max(r.height, 1);
      const fovHalf = THREE.MathUtils.degToRad(cam.fov) / 2;
      const fill = 0.65;
      const z = aspect <= 1
        ? 1 / (fill * Math.tan(fovHalf) * aspect)
        : 1 / (fill * Math.tan(fovHalf));
      cam.position.z = THREE.MathUtils.clamp(z, 2.0, 24);
    }

    function size() {
      const r = canvas!.getBoundingClientRect();
      renderer.setSize(r.width, r.height, false);
      cam.aspect = r.width / Math.max(r.height, 1);
      cam.updateProjectionMatrix();
      fitOrb();
    }

    const uniforms = {
      uTime: { value: 0 },
      uActivity: { value: 0 },
      uBeat: { value: 0 },
      uColorA: { value: new THREE.Color(0xff2fb0) },
      uColorB: { value: new THREE.Color(0x28d8ff) },
      uHot: { value: new THREE.Color(0xffffff) },
    };

    function shell(side: THREE.Side, fv: number) {
      const u = Object.assign({}, uniforms, { uFront: { value: fv } }) as Record<string, THREE.IUniform>;
      return new THREE.ShaderMaterial({
        uniforms: u,
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: false,
        side,
      });
    }

    const geo = new THREE.SphereGeometry(1, 160, 160);
    const back = new THREE.Mesh(geo, shell(THREE.BackSide, 0.0));
    const front = new THREE.Mesh(geo, shell(THREE.FrontSide, 1.0));
    scene.add(back);
    scene.add(front);

    function glowTex() {
      const s = 256;
      const cv = document.createElement("canvas");
      cv.width = cv.height = s;
      const g = cv.getContext("2d")!;
      const r = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
      r.addColorStop(0, "rgba(255,255,255,0.9)");
      r.addColorStop(0.28, "rgba(255,255,255,0.3)");
      r.addColorStop(0.6, "rgba(255,255,255,0.06)");
      r.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = r;
      g.fillRect(0, 0, s, s);
      const tx = new THREE.Texture(cv);
      tx.needsUpdate = true;
      return tx;
    }

    const g1 = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: glowTex(),
        color: 0xff4fc0,
        transparent: true,
        blending: THREE.AdditiveBlending,
        opacity: 0.16,
        depthWrite: false,
      })
    );
    g1.scale.set(3.2, 3.2, 1);
    scene.add(g1);

    let activity = targetRef.current;
    let rot = 0;
    const clock = new THREE.Clock();
    size();

    const ro = new ResizeObserver(size);
    ro.observe(canvas);

    let raf = 0;
    function loop() {
      const dt = clock.getDelta();
      const t = clock.elapsedTime;
      activity += (targetRef.current - activity) * Math.min(dt * 3, 1);
      uniforms.uTime.value = t;
      uniforms.uActivity.value = activity;
      const bpm = 0.9 + activity * 1.4;
      const beat =
        Math.pow(Math.max(0, Math.sin(t * bpm * Math.PI)), 8.0) +
        0.6 * Math.pow(Math.max(0, Math.sin((t * bpm + 0.16) * Math.PI)), 8.0);
      uniforms.uBeat.value = beat * (0.5 + activity * 0.5);
      const s = 1.0 + beat * 0.03 * (1.0 - activity * 0.4) + activity * 0.02;
      back.scale.setScalar(s);
      front.scale.setScalar(s);
      const spin = 1.0 - activity;
      rot += dt * 0.1 * spin;
      const ry = rot;
      const rx = Math.sin(t * 0.13) * 0.1 * spin;
      front.rotation.set(rx, ry, 0);
      back.rotation.set(rx * 0.85, ry * 0.82 + 0.35, 0);
      g1.material.opacity = 0.12 + beat * 0.08 + activity * 0.13;
      cam.lookAt(0, 0, 0);
      renderer.render(scene, cam);
      raf = requestAnimationFrame(loop);
    }
    loop();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      geo.dispose();
      back.material.dispose();
      front.material.dispose();
      g1.material.map?.dispose();
      g1.material.dispose();
      renderer.dispose();
    };
  }, []);

  return <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />;
}

import { useEffect, useState } from 'react';
import type { Capture, Crop } from '../api/types';

/** Assumed capture dimensions until the real image reports its own. */
const FALLBACK_W = 1280;
const FALLBACK_H = 720;

function useImageSize(url: string | undefined): { w: number; h: number } {
  const [size, setSize] = useState({ w: FALLBACK_W, h: FALLBACK_H });
  useEffect(() => {
    if (!url || typeof Image === 'undefined') return;
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (live && img.naturalWidth > 0) {
        setSize({ w: img.naturalWidth, h: img.naturalHeight });
      }
    };
    img.src = url;
    return () => {
      live = false;
    };
  }, [url]);
  return size;
}

/**
 * A magnified window onto the exact pixels a field was read from. The capture
 * image is used as a CSS background and scaled/offset so `crop.box` fills the
 * viewport — no canvas, no second network fetch.
 */
export function CropZoom({
  crop,
  captures,
  width = 190,
  height = 96,
  label,
}: {
  crop: Crop;
  captures: Capture[];
  width?: number;
  height?: number;
  label?: string;
}) {
  const capture = captures.find((c) => c.id === crop.capture_id) ?? captures[0];
  const { w: imgW, h: imgH } = useImageSize(capture?.thumb_url);

  const [bx, by, bw, bh] = crop.box;
  const safeW = bw > 0 ? bw : 1;
  const safeH = bh > 0 ? bh : 1;
  const scale = Math.min(width / safeW, height / safeH);
  const offsetX = -bx * scale + (width - safeW * scale) / 2;
  const offsetY = -by * scale + (height - safeH * scale) / 2;

  if (!capture) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center border border-ink-700 bg-ink-850 text-[10px] text-ink-500"
      >
        no capture
      </div>
    );
  }

  return (
    <figure className="m-0 shrink-0">
      <div
        data-testid="crop-zoom"
        role="img"
        aria-label={label ?? `Zoomed source crop at ${bx},${by}`}
        style={{
          width,
          height,
          backgroundImage: `url("${capture.thumb_url}")`,
          backgroundSize: `${imgW * scale}px ${imgH * scale}px`,
          backgroundPosition: `${offsetX}px ${offsetY}px`,
          backgroundRepeat: 'no-repeat',
          imageRendering: 'pixelated',
        }}
        className="border border-ink-600 bg-ink-950"
      />
      <figcaption className="num mt-0.5 text-[9px] text-ink-500">
        {capture.id} · {Math.round(scale * 100)}% · [{bx},{by},{bw},{bh}]
      </figcaption>
    </figure>
  );
}

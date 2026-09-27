import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';

export function useScoreboardGsap(targetValue: number, options?: { duration?: number; formatFn?: (val: number) => string }) {
  const [displayValue, setDisplayValue] = useState<number>(targetValue);
  const elementRef = useRef<HTMLSpanElement | null>(null);
  const counterObjRef = useRef<{ value: number }>({ value: targetValue });
  const reduceMotionRef = useRef<boolean>(false);

  useEffect(() => {
    const mm = gsap.matchMedia();
    mm.add(
      {
        reduceMotion: '(prefers-reduced-motion: reduce)',
      },
      (context) => {
        reduceMotionRef.current = Boolean(context.conditions?.reduceMotion);
      }
    );

    return () => {
      mm.revert();
    };
  }, []);

  useEffect(() => {
    const startVal = counterObjRef.current.value;
    if (startVal === targetValue) return;

    const reduce = reduceMotionRef.current;
    const dur = reduce ? 0 : (options?.duration ?? 0.35);

    // Number roll tween
    gsap.to(counterObjRef.current, {
      value: targetValue,
      duration: dur,
      ease: 'power2.out',
      overwrite: 'auto',
      onUpdate: () => {
        setDisplayValue(Math.round(counterObjRef.current.value));
      },
    });

    // Subtle micro-bump on update if element is attached
    if (elementRef.current && !reduce) {
      gsap.fromTo(
        elementRef.current,
        { scale: 1.06, color: '#f5f2eb' },
        {
          scale: 1,
          color: '',
          duration: 0.22,
          ease: 'power2.out',
          overwrite: 'auto',
        }
      );
    }
  }, [targetValue, options?.duration]);

  return {
    value: displayValue,
    ref: elementRef,
    formatted: options?.formatFn ? options?.formatFn(displayValue) : displayValue.toString(),
  };
}

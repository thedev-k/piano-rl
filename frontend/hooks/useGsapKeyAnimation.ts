import { useEffect, useRef, useCallback } from 'react';
import gsap from 'gsap';

export type KeyVisualState = 'idle' | 'active' | 'hit' | 'exact' | 'off_by_one' | 'wrong' | 'miss';

export function useGsapKeyAnimation() {
  const keyElementsRef = useRef<Map<number, HTMLDivElement>>(new Map());
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

  const registerKey = useCallback((pitch: number, el: HTMLDivElement | null) => {
    if (el) {
      keyElementsRef.current.set(pitch, el);
    } else {
      keyElementsRef.current.delete(pitch);
    }
  }, []);

  const triggerKeyStrike = useCallback((
    pitch: number,
    state: KeyVisualState,
    isBlack: boolean,
    durationSeconds: number = 0.35
  ) => {
    const el = keyElementsRef.current.get(pitch);
    if (!el) return;

    const reduce = reduceMotionRef.current;
    const depressionY = isBlack ? 2 : 3;

    if (state === 'idle') {
      gsap.to(el, {
        y: 0,
        duration: reduce ? 0 : 0.18,
        ease: 'power2.out',
        overwrite: 'auto',
      });
      return;
    }

    if (state === 'wrong') {
      // Mechanical friction vibration / jarring strike for wrong key
      if (reduce) {
        gsap.to(el, { y: 0, duration: 0, overwrite: 'auto' });
      } else {
        const tl = gsap.timeline({ overwrite: 'auto' });
        tl.to(el, {
          y: depressionY + 0.5,
          x: -1,
          duration: 0.05,
          ease: 'power3.out',
        })
          .to(el, {
            x: 1,
            duration: 0.05,
            ease: 'power2.inOut',
          })
          .to(el, {
            x: 0,
            y: depressionY,
            duration: 0.06,
          })
          .to(el, {
            y: 0,
            duration: 0.22,
            ease: 'power2.out',
            delay: Math.max(0.08, durationSeconds - 0.2),
          });
      }
      return;
    }

    // Accurate hit or active strike: crisp physical hammer depression and elastic rebound
    if (reduce) {
      gsap.to(el, {
        y: 0,
        duration: 0,
        overwrite: 'auto',
      });
    } else {
      const tl = gsap.timeline({ overwrite: 'auto' });
      tl.to(el, {
        y: depressionY,
        scaleY: isBlack ? 0.99 : 0.995,
        transformOrigin: 'top center',
        duration: 0.06,
        ease: 'power2.out',
      }).to(el, {
        y: 0,
        scaleY: 1,
        duration: 0.2,
        ease: 'power1.out',
        delay: Math.max(0.06, durationSeconds - 0.15),
      });
    }
  }, []);

  return {
    registerKey,
    triggerKeyStrike,
  };
}

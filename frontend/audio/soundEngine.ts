import * as Tone from 'tone';

class SoundEngine {
  private sampler: Tone.Sampler | null = null;
  private isLoaded: boolean = false;
  private isStarted: boolean = false;
  private activeNotes: Set<string> = new Set();
  private isPedalDown: boolean = false;

  constructor() {
    this.initSampler();
  }

  private initSampler() {
    try {
      this.sampler = new Tone.Sampler({
        urls: {
          A0: 'A0.mp3',
          C1: 'C1.mp3',
          'D#1': 'Ds1.mp3',
          'F#1': 'Fs1.mp3',
          A1: 'A1.mp3',
          C2: 'C2.mp3',
          'D#2': 'Ds2.mp3',
          'F#2': 'Fs2.mp3',
          A2: 'A2.mp3',
          C3: 'C3.mp3',
          'D#3': 'Ds3.mp3',
          'F#3': 'Fs3.mp3',
          A3: 'A3.mp3',
          C4: 'C4.mp3',
          'D#4': 'Ds4.mp3',
          'F#4': 'Fs4.mp3',
          A4: 'A4.mp3',
          C5: 'C5.mp3',
          'D#5': 'Ds5.mp3',
          'F#5': 'Fs5.mp3',
          A5: 'A5.mp3',
          C6: 'C6.mp3',
          'D#6': 'Ds6.mp3',
          'F#6': 'Fs6.mp3',
          A6: 'A6.mp3',
          C7: 'C7.mp3',
          'D#7': 'Ds7.mp3',
          'F#7': 'Fs7.mp3',
          A7: 'A7.mp3',
          C8: 'C8.mp3',
        },
        baseUrl: '/audio/salamander/',
        onload: () => {
          this.isLoaded = true;
        },
      }).toDestination();
      this.sampler.volume.value = -4; // Subdued, acoustic-grade studio level
    } catch (err) {
      console.warn('Tone.js Sampler initialization fallback:', err);
    }
  }

  public async startAudioContext(): Promise<boolean> {
    if (this.isStarted) return true;
    try {
      await Tone.start();
      this.isStarted = true;
      return true;
    } catch (err) {
      console.warn('Failed to start Tone.js audio context:', err);
      return false;
    }
  }

  public get audioReady(): boolean {
    return this.isStarted && this.isLoaded;
  }

  public setPedal(down: boolean) {
    this.isPedalDown = down;
    if (!down && this.sampler) {
      // When pedal lifts, release sustained notes that were queued for release
      for (const note of this.activeNotes) {
        this.sampler.triggerRelease(note, Tone.now());
      }
      this.activeNotes.clear();
    }
  }

  public playNote(pitch: number, durationSeconds: number = 0.5, velocity: number = 0.75) {
    if (!this.sampler || !this.isLoaded) return;
    try {
      const noteName = Tone.Frequency(pitch, 'midi').toNote();
      const now = Tone.now();
      const clampedVelocity = Math.max(0.1, Math.min(1.0, velocity));
      const dur = Math.max(0.08, durationSeconds);

      this.sampler.triggerAttack(noteName, now, clampedVelocity);
      this.activeNotes.add(noteName);

      if (!this.isPedalDown) {
        this.sampler.triggerRelease(noteName, now + dur);
        setTimeout(() => {
          this.activeNotes.delete(noteName);
        }, dur * 1000);
      }
    } catch (e) {
      console.warn('Error playing note:', pitch, e);
    }
  }

  public stopAll() {
    if (!this.sampler) return;
    try {
      this.sampler.releaseAll();
      this.activeNotes.clear();
    } catch {
      // Ignored
    }
  }
}

export const soundEngine = new SoundEngine();

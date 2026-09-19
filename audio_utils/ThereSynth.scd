s.options.outDevice = "DELL S2421H (NVIDIA High Definition Audio)";

s.waitForBoot {
    SynthDef(\thereSynth, { |freq = 0.5, amp = 0.2, gate = 1, frame=0.3, verb=0.1|
        var sig, env, freq_lin, ampli;

        env = EnvGen.kr(Env.asr(0.01, 1, 0.2), gate, doneAction: 2);

		// ZED RANGES
        //freq_lin = LinLin.kr(freq, 0.55, 0.85, 0.0, 1.0);
		//ampli = LinLin.kr(amp.clip(0.05, 1.0), 0.05, 0.15, 0.0, 1.0);

		// MOCAP RANGES
        freq_lin = LinLin.kr(freq, 0.45, 0.65, 0.0, 1.0);
		ampli = LinLin.kr(amp.clip(0.05, 1.0), 0.05, 0.15, 0.0, 1.0);

        sig = SinOsc.ar(freq_lin.lag(frame)*2000) * ampli.lag(frame) * env * 2;
		sig = FreeVerb.ar(sig, mix: verb, room:0.3, damp: 0.30);

        Out.ar(0, Limiter.ar(sig!2, 0.9));
    }).add;

    s.sync;

    ~theremin = Synth.new(\thereSynth);

    OSCdef(\pitchCtrl, { |msg|
        var freq = msg[1].asFloat;
        ~theremin.set(\freq, freq);
    }, '/pitch');

    OSCdef(\volumeCtrl, { |msg|
        var amp = msg[1].asFloat;
        ~theremin.set(\amp, amp);
    }, '/volume');
};

OSCFunc.trace(false);

NetAddr.langPort

~theremin.set(\freq, 0.2)

ServerOptions.outDevices
/**
 * pose.js — Browser-side MediaPipe Pose detection.
 *
 * Exposes:
 *   PoseTracker.init(videoEl, canvasEl, labelEl) → Promise<void>
 *   PoseTracker.getPoseSnapshot()                → string
 *   PoseTracker.setActive(bool)                  → void  (pause/resume updating snapshot)
 */

const PoseTracker = (() => {
  let currentPose = "standing naturally";
  let active = true; // update snapshot only while active

  // -----------------------------------------------------------------------
  // Landmark indices (MediaPipe Pose 33-point model)
  // -----------------------------------------------------------------------
  const LM = {
    NOSE:           0,
    LEFT_SHOULDER:  11,
    RIGHT_SHOULDER: 12,
    LEFT_ELBOW:     13,
    RIGHT_ELBOW:    14,
    LEFT_WRIST:     15,
    RIGHT_WRIST:    16,
    LEFT_HIP:       23,
    RIGHT_HIP:      24,
    LEFT_KNEE:      25,
    RIGHT_KNEE:     26,
    LEFT_ANKLE:     27,
    RIGHT_ANKLE:    28,
  };

  // -----------------------------------------------------------------------
  // interpretPose — convert 33 landmarks to a human-readable descriptor
  // -----------------------------------------------------------------------
  function interpretPose(landmarks) {
    const lm = landmarks;
    const descriptors = [];

    const lw = lm[LM.LEFT_WRIST];
    const rw = lm[LM.RIGHT_WRIST];
    const ls = lm[LM.LEFT_SHOULDER];
    const rs = lm[LM.RIGHT_SHOULDER];
    const le = lm[LM.LEFT_ELBOW];
    const re = lm[LM.RIGHT_ELBOW];
    const lh = lm[LM.LEFT_HIP];
    const rh = lm[LM.RIGHT_HIP];
    const lk = lm[LM.LEFT_KNEE];
    const rk = lm[LM.RIGHT_KNEE];
    const la = lm[LM.LEFT_ANKLE];
    const ra = lm[LM.RIGHT_ANKLE];
    const nose = lm[LM.NOSE];

    const leftArmUp  = lw.y < ls.y - 0.05;
    const rightArmUp = rw.y < rs.y - 0.05;

    // 1. Both arms raised
    if (leftArmUp && rightArmUp) {
      descriptors.push("arms raised triumphantly in the air");
    } else {
      // 2. Single arm raised
      if (leftArmUp)  descriptors.push("left arm raised high");
      if (rightArmUp) descriptors.push("right arm raised high");
    }

    // 3. Wide arm span (T-pose / wings)
    const armSpan = Math.abs(lw.x - rw.x);
    if (armSpan > 0.65) {
      descriptors.push("arms stretched wide open like wings");
    }

    // 4. One arm pointing up, other at side → pointing gesture
    if (leftArmUp && !rightArmUp && Math.abs(rw.x - rs.x) < 0.15) {
      descriptors.push("pointing upward with the left hand");
    } else if (rightArmUp && !leftArmUp && Math.abs(lw.x - ls.x) < 0.15) {
      descriptors.push("pointing upward with the right hand");
    }

    // 5. Crouching — hips are close to nose level in y
    const hipMidY = (lh.y + rh.y) / 2;
    if (Math.abs(hipMidY - nose.y) < 0.22) {
      descriptors.push("crouching down low");
    }

    // 6. Leaning left/right — nose offset from hip midpoint
    const hipMidX = (lh.x + rh.x) / 2;
    const leanOffset = nose.x - hipMidX;
    if (leanOffset < -0.15) descriptors.push("leaning to the left");
    else if (leanOffset > 0.15) descriptors.push("leaning to the right");

    // 7. Arms crossed — left wrist is right of right wrist (or vice-versa)
    if (lw.x > rw.x + 0.1) {
      descriptors.push("arms crossed");
    }

    // 8. Hands on hips — wrists near hip level and close to body midline
    const bodyMidX = (ls.x + rs.x) / 2;
    const lwNearHip = Math.abs(lw.y - lh.y) < 0.12 && Math.abs(lw.x - bodyMidX) < 0.3;
    const rwNearHip = Math.abs(rw.y - rh.y) < 0.12 && Math.abs(rw.x - bodyMidX) < 0.3;
    if (lwNearHip && rwNearHip) {
      descriptors.push("hands on hips confidently");
    }

    // 9. Jumping — both ankles above hip level (y-coords inverted: lower value = higher up)
    if (la.y < lh.y - 0.05 && ra.y < rh.y - 0.05) {
      descriptors.push("jumping in the air");
    }

    // 10. Kneeling — one knee is near the ground (high y value)
    if (lk.y > lh.y + 0.3 || rk.y > rh.y + 0.3) {
      descriptors.push("kneeling down on one knee");
    }

    // 11. Waving — one wrist near head level and moving
    if ((lw.y < nose.y + 0.1 && lw.y > nose.y - 0.2) && !leftArmUp) {
      descriptors.push("waving with the left hand");
    } else if ((rw.y < nose.y + 0.1 && rw.y > nose.y - 0.2) && !rightArmUp) {
      descriptors.push("waving with the right hand");
    }

    // Default
    if (descriptors.length === 0) {
      return "standing naturally";
    }

    return descriptors.join(", ");
  }

  // -----------------------------------------------------------------------
  // drawPoseSkeleton — lightweight canvas drawing using drawing_utils
  // -----------------------------------------------------------------------
  function drawPoseSkeleton(canvasCtx, landmarks) {
    canvasCtx.clearRect(0, 0, canvasCtx.canvas.width, canvasCtx.canvas.height);

    // Use MediaPipe's drawing utils if available, otherwise draw manually
    if (window.drawConnectors && window.drawLandmarks && window.POSE_CONNECTIONS) {
      drawConnectors(canvasCtx, landmarks, POSE_CONNECTIONS, {
        color: "rgba(255, 215, 0, 0.7)",
        lineWidth: 3,
      });
      drawLandmarks(canvasCtx, landmarks, {
        color: "rgba(255, 100, 50, 0.9)",
        lineWidth: 1,
        radius: 4,
      });
    } else {
      // Fallback: draw simple dots
      landmarks.forEach((lm) => {
        if (lm.visibility < 0.5) return;
        const x = lm.x * canvasCtx.canvas.width;
        const y = lm.y * canvasCtx.canvas.height;
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 5, 0, Math.PI * 2);
        canvasCtx.fillStyle = "rgba(255, 215, 0, 0.85)";
        canvasCtx.fill();
      });
    }
  }

  // -----------------------------------------------------------------------
  // init — set up MediaPipe Pose and attach to webcam
  // -----------------------------------------------------------------------
  async function init(videoEl, canvasEl, labelEl) {
    const canvasCtx = canvasEl.getContext("2d");

    const pose = new Pose({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    pose.onResults((results) => {
      // Sync canvas size to video
      canvasEl.width  = videoEl.videoWidth  || 640;
      canvasEl.height = videoEl.videoHeight || 480;

      if (!results.poseLandmarks) {
        canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        if (labelEl) labelEl.textContent = "No pose detected";
        return;
      }

      drawPoseSkeleton(canvasCtx, results.poseLandmarks);

      const desc = interpretPose(results.poseLandmarks);
      if (active) currentPose = desc;
      if (labelEl) labelEl.textContent = desc;
    });

    // Use MediaPipe Camera util to drive the pose model frame-by-frame
    const camera = new Camera(videoEl, {
      onFrame: async () => {
        await pose.send({ image: videoEl });
      },
      width: 640,
      height: 480,
    });

    await camera.start();
    console.log("[PoseTracker] Camera and pose model started.");
  }

  function getPoseSnapshot() {
    return currentPose;
  }

  function setActive(flag) {
    active = flag;
  }

  return { init, getPoseSnapshot, setActive };
})();

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

IDX = {"left_shoulder":11,"right_shoulder":12,"left_elbow":13,"right_elbow":14,
       "left_wrist":15,"right_wrist":16,"left_hip":23,"right_hip":24,
       "left_knee":25,"right_knee":26,"left_ankle":27,"right_ankle":28}

def angle(a,b,c):
    a,b,c = np.asarray(a,float),np.asarray(b,float),np.asarray(c,float)
    ba,bc=a-b,c-b
    den=np.linalg.norm(ba)*np.linalg.norm(bc)
    if den < 1e-8: return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/den,-1,1))))

def extract_video_features(video_path,max_frames=160,min_visibility=0.35,make_overlay=False):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {video_path}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration=n/fps if n>0 else np.nan
    sample_ids=set(np.linspace(0,max(n-1,0),min(max_frames,n),dtype=int)) if n>0 else None
    rows=[]; overlay=None; i=0
    pose=mp.solutions.pose
    with pose.Pose(static_image_mode=False,model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5) as model:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if sample_ids is not None and i not in sample_ids:
                i+=1; continue
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=model.process(rgb)
            row={"time_sec":i/fps,"pose_detected":0,"visible_landmark_fraction":0.0}
            if res.pose_landmarks:
                lm=res.pose_landmarks.landmark
                row["pose_detected"]=1
                row["visible_landmark_fraction"]=float(np.mean([p.visibility>=min_visibility for p in lm]))
                pts={}
                for name,idx in IDX.items():
                    p=lm[idx]; pts[name]=(p.x,p.y)
                    row[name+"_x"]=p.x; row[name+"_y"]=p.y; row[name+"_visibility"]=p.visibility
                for side in ["left","right"]:
                    row[side+"_elbow_angle"]=angle(pts[side+"_shoulder"],pts[side+"_elbow"],pts[side+"_wrist"])
                    row[side+"_knee_angle"]=angle(pts[side+"_hip"],pts[side+"_knee"],pts[side+"_ankle"])
                    row[side+"_hip_angle"]=angle(pts[side+"_shoulder"],pts[side+"_hip"],pts[side+"_knee"])
                row["shoulder_line_tilt_deg"]=float(np.degrees(np.arctan2(
                    pts["right_shoulder"][1]-pts["left_shoulder"][1],
                    pts["right_shoulder"][0]-pts["left_shoulder"][0]+1e-8)))
                if make_overlay and overlay is None:
                    overlay=rgb.copy()
                    mp.solutions.drawing_utils.draw_landmarks(overlay,res.pose_landmarks,pose.POSE_CONNECTIONS)
            rows.append(row); i+=1
    cap.release()
    ts=pd.DataFrame(rows)
    if ts.empty: raise ValueError("No frames processed. Try an MP4 video.")
    features={"duration_sec":float(duration) if np.isfinite(duration) else float(ts.time_sec.max()),
              "sampled_frames":int(len(ts)),
              "pose_detection_rate":float(ts.pose_detected.mean()),
              "mean_visible_landmark_fraction":float(ts.visible_landmark_fraction.mean())}
    for col in ts.select_dtypes(include=[np.number]).columns:
        if col in ["time_sec","pose_detected","visible_landmark_fraction"] or col.endswith("_visibility"): continue
        vals=ts[col].replace([np.inf,-np.inf],np.nan).dropna()
        if len(vals):
            features[col+"_mean"]=float(vals.mean())
            features[col+"_std"]=float(vals.std(ddof=0)) if len(vals)>1 else 0.0
            if col.endswith("_angle"):
                features[col+"_min"]=float(vals.min()); features[col+"_max"]=float(vals.max())
                features[col+"_range"]=float(vals.max()-vals.min())
    return features,overlay,ts

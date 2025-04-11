import cv2
import numpy as np
import os

# Checkerboard settings
CHECKERBOARD = (6, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chuẩn bị điểm 3D thật sự
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)

# Danh sách để lưu các điểm
objpoints = []      # 3D points trong không gian thực
imgpointsL = []     # 2D points từ camera trái
imgpointsR = []     # 2D points từ camera phải

# Khởi tạo camera
camL = cv2.VideoCapture(2, cv2.CAP_DSHOW)
camR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Press 's' to save frame pair, 'r' to run stereo calibration, 'q' to quit.")

frame_idx = 0
while True:
    retL, frameL = camL.read()
    retR, frameR = camR.read()
    if not retL or not retR:
        print("❌ Không đọc được camera")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    # Nếu tìm được trên cả 2 ảnh
    if retL and retR:
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(frameL, CHECKERBOARD, cornersL, retL)
        cv2.drawChessboardCorners(frameR, CHECKERBOARD, cornersR, retR)

    combined = cv2.hconcat([frameL, frameR])
    cv2.imshow('Stereo Calibration', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        print(f"✅ Lưu frame pair #{len(objpoints)}")
    elif key == ord('r'):
        if len(objpoints) < 5:
            print("⚠️ Cần ít nhất 5 frame pair để calibration.")
            continue

        print("🔧 Running stereo calibration...")

        # Calibrate từng camera trước
        retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
        retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR,
            mtxL, distL, mtxR, distR,
            grayL.shape[::-1], None, None, None, None,
            flags=flags, criteria=criteria
        )

        # Stereo Rectify
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            mtxL, distL, mtxR, distR,
            grayL.shape[::-1], R, T
        )

        print("\n🎯 Stereo Calibration Results:")
        print(f"Camera Matrix Left:\n{mtxL}")
        print(f"Camera Matrix Right:\n{mtxR}")
        print(f"Distortion Left:\n{distL}")
        print(f"Distortion Right:\n{distR}")
        print(f"Rotation Matrix R:\n{R}")
        print(f"Translation Vector T:\n{T}")
        print(f"Q Matrix:\n{Q}")

        # Save calibration
        np.savez("stereo_calib_data.npz", mtxL=mtxL, distL=distL,
                 mtxR=mtxR, distR=distR, R=R, T=T, Q=Q)
        print("📁 Saved calibration to 'stereo_calib_data.npz'")

camL.release()
camR.release()
cv2.destroyAllWindows()

# Lưu file .yml
fs = cv2.FileStorage("stereo_calibration.yml", cv2.FILE_STORAGE_WRITE)

fs.write("CameraMatrixL", mtxL)
fs.write("DistCoeffsL", distL)
fs.write("CameraMatrixR", mtxR)
fs.write("DistCoeffsR", distR)
fs.write("RotationMatrix", R)
fs.write("TranslationVector", T)
fs.write("QMatrix", Q)

fs.release()
print("✅ Đã lưu file stereo_calibration.yml")

import cv2
import numpy as np
import os


class CameraCalibrator:
    def __init__(self, folder, size, square, target=(800, 800), show=True, opencv=True):
        self.folder = folder
        self.size = size
        self.square = square
        self.target = target
        self.show = show
        self.opencv = opencv
        self.objp = np.zeros((size[0] * size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
        self.objp *= square
        self.objpoints = []
        self.imgpoints = []

    def find_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.findChessboardCorners(gray, self.size,
                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    def calibrate_opencv(self, images):
        detected = 0
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            img = cv2.resize(img, self.target, interpolation=cv2.INTER_AREA)
            ret, corners = self.find_corners(img)
            if ret:
                detected += 1
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                if self.show:
                    cv2.drawChessboardCorners(img, self.size, corners, ret)
                    cv2.imshow('Corners', img)
                    cv2.waitKey(500)
        cv2.destroyAllWindows()

        if detected == 0:
            return

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.target, None, None)
        if not ret:
            return

        print("\n=== Intrinsic Parameters (Camera Matrix) ===\n", mtx)

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            print(f"Rotation matrix for image {i + 1}:\n", R)
            print(f"Translation vector for image {i + 1}:\n", tvec)

        return mtx, dist, rvecs, tvecs

    def calibrate_manual(self, images):
        detected = 0
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            img = cv2.resize(img, self.target, interpolation=cv2.INTER_AREA)
            ret, corners = self.find_corners(img)
            if ret:
                detected += 1
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                if self.show:
                    cv2.drawChessboardCorners(img, self.size, corners, ret)
                    cv2.imshow('Corners', img)
                    cv2.waitKey(500)
        cv2.destroyAllWindows()

        if detected == 0:
            return

        imgpoints = np.array(self.imgpoints, dtype=np.float32)
        objpoints = np.array(self.objpoints, dtype=np.float32)
        h, w = self.target
        K = np.eye(3)
        K[0, 2] = w / 2
        K[1, 2] = h / 2
        focal = 1
        K[0, 0] = focal
        K[1, 1] = focal

        rvecs = []
        tvecs = []

        for i, corners in enumerate(self.imgpoints):
            ret, rvec, tvec = cv2.solvePnP(self.objpoints[i], corners, K, None)
            if ret:
                rvecs.append(rvec)
                tvecs.append(tvec)
                R, _ = cv2.Rodrigues(rvec)
                print(f"Rotation matrix for image {i + 1}:\n", R)
                print(f"Translation vector for image {i + 1}:\n", tvec)

        print("\n=== Intrinsic Parameters (Camera Matrix) ===\n", K)
        print("\nNo distortion coefficients")

        return K, rvecs, tvecs

    def calibrate(self):
        images = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.endswith(('.jpg', '.png'))]

        if self.opencv:
            return self.calibrate_opencv(images)
        else:
            return self.calibrate_manual(images)


if __name__ == "__main__":
    print("Create by Etcharla Revanth Rao and Swarnava Bose")

    folder = "images"
    size = (7, 6)
    square = 10
    opencv = False

    calibrator = CameraCalibrator(folder, size, square, opencv=opencv)
    calibrator.calibrate()

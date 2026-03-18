import sys

def main():
    with open("Analysis/AnalysisOfflineGUI.py", "r") as f:
        content = f.read()

    # --- Part 1: Mask Auto-Guessing Loop ---
    start_str_1 = """                                    image = event.cam_data['image2D']
                                    if image.shape[0] <= 2:
                                        found_masked = True
                                        break

                                    center_region = image[154:454, 284:684]"""

    replacement_1 = """                                    is_1d_profiles = False
                                    image_2d = None

                                    if isinstance(event.cam_data, list) and len(event.cam_data) >= 2:
                                        is_1d_profiles = True
                                    elif isinstance(event.cam_data, dict) and 'image2D' in event.cam_data:
                                        img_data = event.cam_data['image2D']
                                        if isinstance(img_data, list) and len(img_data) >= 2:
                                            is_1d_profiles = True
                                        elif hasattr(img_data, 'shape') and img_data.shape[0] <= 2:
                                            is_1d_profiles = True
                                        else:
                                            image_2d = img_data

                                    if is_1d_profiles or image_2d is None:
                                        found_masked = True
                                        break

                                    center_region = image_2d[154:454, 284:684]"""
    content = content.replace(start_str_1, replacement_1)

    # --- Part 2: Main Processing Loop ---
    start_str_2 = """                    image = event.cam_data['image2D']"""
    end_str_2 = """                    current_cpc = cpc_per_bunch.get(delaySlots, global_cpc_fallback)"""

    start_idx = content.find(start_str_2)
    end_idx = content.find(end_str_2, start_idx) + len(end_str_2)

    replacement_2 = """                    is_1d_profiles = False
                    profile_h, profile_v = None, None
                    image_2d = None

                    if isinstance(event.cam_data, list) and len(event.cam_data) >= 2:
                        is_1d_profiles = True
                        profile_h, profile_v = event.cam_data[0], event.cam_data[1]
                    elif isinstance(event.cam_data, dict) and 'image2D' in event.cam_data:
                        img_data = event.cam_data['image2D']
                        if isinstance(img_data, list) and len(img_data) >= 2:
                            is_1d_profiles = True
                            profile_h, profile_v = img_data[0], img_data[1]
                        elif hasattr(img_data, 'shape') and img_data.shape[0] <= 2:
                            is_1d_profiles = True
                            profile_h = img_data[0]
                            if img_data.shape[0] > 1:
                                profile_v = img_data[1]
                            else:
                                profile_v = np.zeros_like(profile_h)
                        else:
                            image_2d = img_data

                    if not is_1d_profiles and image_2d is None:
                        continue # Unrecognized data format

                    exposure_raw = event.device_data.get('GATE-INT', {}).get('numberOfPulses', 1)
                    exposure = int(np.asarray(exposure_raw).item()) if isinstance(exposure_raw, np.ndarray) else int(exposure_raw)
                    if exposure == 0: exposure = 1

                    fArr = event.device_data.get('FILTERS.B2', {}).get('channelState', [0,0,0])
                    # Ensure scalars for the math
                    fArr0 = np.asarray(fArr[0]).item() if isinstance(fArr[0], np.ndarray) else fArr[0]
                    fArr1 = np.asarray(fArr[1]).item() if isinstance(fArr[1], np.ndarray) else fArr[1]
                    fArr2 = np.asarray(fArr[2]).item() if isinstance(fArr[2], np.ndarray) else fArr[2]
                    in_filter = pow(12.6, 3 * fArr0 + 2 * fArr1 + fArr2)

                    cmask_x = event.device_data.get('CMASK-X', None)
                    cmask_y = event.device_data.get('CMASK-Y', None)

                    # Assuming mask is present if CMASK is close to mask position
                    # "different CMASK-X or Y value" for background
                    # Let's say if abs(cmask_x - maskX) < some_threshold, it's masked
                    is_bg_H = False
                    if cmask_x is not None:
                        cmask_x_val = np.asarray(cmask_x).item() if isinstance(cmask_x, np.ndarray) else cmask_x
                        if abs(cmask_x_val - maskX) > 100:
                            is_bg_H = True
                    # If there's no CMASK info, assume masked for safety

                    is_bg_V = False
                    if cmask_y is not None:
                        cmask_y_val = np.asarray(cmask_y).item() if isinstance(cmask_y, np.ndarray) else cmask_y
                        if abs(cmask_y_val - maskY) > 100:
                            is_bg_V = True

                    if is_1d_profiles:
                        projection_x = np.asarray(profile_h)
                        projection_x = projection_x - np.mean(projection_x[:50])

                        projection_y = np.asarray(profile_v)
                        projection_y = projection_y - np.mean(projection_y[:50])

                        # Calculate CPC if background image
                        if is_bg_H or is_bg_V:
                            # if 1D profile, img_sum approximation
                            img_sum = np.sum(projection_x[264:644]) + np.sum(projection_y[154:454])
                            if img_sum > 0:
                                current_cpc = (bunch_charge * exposure) / (img_sum * INTENSIFIER_GAIN * in_filter)
                                cpc_per_bunch[delaySlots] = current_cpc
                                global_cpc_fallback = current_cpc # update fallback
                    else:
                        projection_x = np.sum(image_2d[y1:y2, :], axis=0)
                        projection_x = projection_x - np.mean(projection_x[:50])

                        projection_y = np.sum(image_2d[:, x1:x2], axis=1)
                        projection_y = projection_y - np.mean(projection_y[:50])

                        # Calculate CPC if background image
                        if is_bg_H or is_bg_V:
                            img_sum = np.sum(image_2d[y1:y2, 264:644])
                            if img_sum > 0:
                                current_cpc = (bunch_charge * exposure) / (img_sum * INTENSIFIER_GAIN * in_filter)
                                cpc_per_bunch[delaySlots] = current_cpc
                                global_cpc_fallback = current_cpc # update fallback

                    current_cpc = cpc_per_bunch.get(delaySlots, global_cpc_fallback)"""

    new_content = content[:start_idx] + replacement_2 + content[end_idx:]

    # Also need to make sure MaskValidationDialog uses image_2d
    new_content = new_content.replace(
        "dialog = MaskValidationDialog(image, guessedX, guessedY, maskR, self)",
        "dialog = MaskValidationDialog(image_2d, guessedX, guessedY, maskR, self)"
    )

    with open("Analysis/AnalysisOfflineGUI.py", "w") as f:
        f.write(new_content)

if __name__ == "__main__":
    main()

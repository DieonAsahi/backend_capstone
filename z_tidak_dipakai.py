
# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('index'))


# @app.route('/wardrobe')
# def wardrobe_page():
#     if not session.get("user_id"):
#         return redirect(url_for('index'))
#     return render_template('wardrobe.html')


# @app.route('/predict_image', methods=['POST'])
# def predict_image():
#     """API BARU: Menerima foto, menjalankan AI, mengembalikan prediksi."""
#     if not session.get("user_id"): return jsonify(error="Not logged in"), 401
#     if not AI_MODE_AKTIF: return jsonify(error="AI mode is not active"), 500
        
#     data = request.get_json()
#     image_data_url = data.get('image_data')
    
#     # Simpan gambar sementara untuk dianalisis
#     try:
#         header, encoded = image_data_url.split(",", 1)
#         image_data = base64.b64decode(encoded)
#         temp_filename = f"temp_{session.get('user_id')}.jpg"
#         temp_image_path = os.path.join(UPLOAD_DIR, temp_filename)
        
#         with open(temp_image_path, "wb") as f:
#             f.write(image_data)
            
#         # Panggil "otak" AI kita
#         models_dict = {'style': model_style, 'men': model_men, 'women': model_women}
#         gender = session.get("gender")
        
#         predictions = predict_logic.run_all_predictions(
#             temp_image_path, gender, models_dict, VALID_RULES, MODEL_CLASSES
#         )
        
#         # Hapus file sementara
#         os.remove(temp_image_path)
        
#         return jsonify(predictions)
        
#     except Exception as e:
#         print(f"Error saat prediksi AI: {e}")
#         return jsonify(style="Error", kategori="AI", warna="Gagal"), 500

# @app.route('/get_categories', methods=['POST'])
# def get_categories():
#     # (Kode SAMA seperti sebelumnya)
#     if not session.get("user_id"): return jsonify(error="Not logged in"), 401
#     data = request.get_json()
#     style = data.get('style')
#     gender = session.get("gender")
#     try:
#         categories = VALID_RULES[gender][style]
#         return jsonify(categories=categories)
#     except KeyError:
#         return jsonify(categories=[]), 404

# @app.route('/save_item', methods=['POST'])
# def save_item():
#     # (Kode SAMA seperti sebelumnya, menggunakan flask_mysqldb)
#     if not session.get("user_id"): return jsonify(error="Not logged in"), 401
#     data = request.get_json()
#     style = data.get('style')
#     category_name = data.get('category')
#     color_name = data.get('color')
#     image_data_url = data.get('image_data')
    
#     # 1. Simpan Gambar Base64 ke File
#     try:
#         header, encoded = image_data_url.split(",", 1)
#         image_data = base64.b64decode(encoded)
#         filename = str(uuid.uuid4()) + ".jpg"
#         image_path = os.path.join(UPLOAD_DIR, filename)
#         with open(image_path, "wb") as f: f.write(image_data)
#         image_db_url = f"static/uploads/{filename}"
#     except Exception as e:
#         print(f"Error menyimpan gambar: {e}")
#         return jsonify(status='error', message="Gagal memproses gambar"), 500

#     # 2. Simpan Info ke Database
#     try:
#         cur = mysql.connection.cursor()
        
#         # Cari/Buat Kategori
#         cur.execute("SELECT * FROM categories WHERE category_name = %s", [category_name])
#         category = cur.fetchone()
#         category_id = category['category_id'] if category else None
#         if not category:
#             cur.execute("INSERT INTO categories (category_name) VALUES (%s)", [category_name])
#             mysql.connection.commit()
#             category_id = cur.lastrowid

#         # Cari/Buat Warna
#         cur.execute("SELECT * FROM colors WHERE color_name = %s", [color_name])
#         color = cur.fetchone()
#         color_id = color['color_id'] if color else None
#         if not color:
#             cur.execute("INSERT INTO colors (color_name) VALUES (%s)", [color_name])
#             mysql.connection.commit()
#             color_id = cur.lastrowid
            
#         # Masukkan ke user_wardrobe
#         cur.execute(
#             """INSERT INTO user_wardrobe (user_id, item_name, image_url, category_id, color_id, style) 
#                VALUES (%s, %s, %s, %s, %s, %s)""",
#             (session.get("user_id"), f"{style} {category_name} {color_name}", 
#              image_db_url, category_id, color_id, style)
#         )
#         mysql.connection.commit()
#         cur.close()
#         return jsonify(status='success', message=f'Item berhasil disimpan!')
#     except Exception as e:
#         mysql.connection.rollback()
#         cur.close()
#         print(f"Error saat menyimpan ke DB: {e}")
#         return jsonify(status='error', message=str(e)), 500

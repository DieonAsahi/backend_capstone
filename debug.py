@app.route('/api/debug/db-structure', methods=['GET'])
def debug_db_structure():
    if not session.get("user_id"):
        return jsonify(error="Not logged in"), 401

    tables = ['skin_tones','body_shapes','users','categories', 'colors', 'user_wardrobe']
    result = {}

    try:
        cur = mysql.connection.cursor()
        for table in tables:
            cur.execute(f"DESCRIBE {table}")
            result[table] = cur.fetchall()
        cur.close()

        return jsonify(success=True, tables=result)

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


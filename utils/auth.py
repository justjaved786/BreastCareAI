import sqlite3
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import re
import json

# ----------------------------
# Database Configuration
# ----------------------------

DATABASE = 'users.db'

def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def create_user_table():
    """Create the users table in the database if it doesn't exist with the updated schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date_of_birth DATE,
            gender TEXT,
            phone TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_user_history_table():
    """Create the user_history table for tracking user predictions and activities."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_type TEXT NOT NULL,
            model_used TEXT,
            result TEXT,
            confidence REAL,
            input_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create index for better performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_history_user_id 
        ON user_history (user_id)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_history_created_at 
        ON user_history (created_at)
    ''')
    
    conn.commit()
    conn.close()

def init_database():
    """Initialize the database with the updated schema."""
    create_user_table()
    create_user_history_table()

# ----------------------------
# User Registration
# ----------------------------

def register_user(name, email, password, date_of_birth=None, gender=None):
    """Register a new user in the database with proper password hashing - role parameter removed."""
    try:
        # Validate input
        if not name or not email or not password:
            raise ValueError("All fields are required")
        
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Enhanced password validation
        if not any(char.isupper() for char in password):
            raise ValueError("Password must contain at least one uppercase letter")
            
        if not any(char.islower() for char in password):
            raise ValueError("Password must contain at least one lowercase letter")
            
        if not any(char.isdigit() for char in password):
            raise ValueError("Password must contain at least one number")
            
        if not any(not char.isalnum() for char in password):
            raise ValueError("Password must contain at least one special character")
        
        # Email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("Please enter a valid email address")
        
        # Hash the password using werkzeug
        password_hash = generate_password_hash(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Role is hardcoded to 'user' and removed from parameters
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, date_of_birth, gender, role) VALUES (?, ?, ?, ?, ?, ?)",
                (name.strip(), email.strip().lower(), password_hash, date_of_birth, gender, 'user')  # Role hardcoded
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            raise ValueError("Email already exists")
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")
        finally:
            conn.close()
            
    except Exception as e:
        raise Exception(f"Registration failed: {str(e)}")

# ----------------------------
# Login Validation
# ----------------------------

def validate_login(email, password):
    """Validate user login credentials using werkzeug password checking."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),)
        )
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            return dict(user)  # Convert to dict for easier handling
        return None
        
    except Exception as e:
        print(f"Login validation error: {str(e)}")
        return None

# ----------------------------
# User Management
# ----------------------------

def check_email_exists(email):
    """Check if a user with the given email exists."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email = ?", 
            (email.strip().lower(),)
        )
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    except Exception as e:
        print(f"Error checking email: {str(e)}")
        return None

def reset_user_password(email, new_password):
    """Reset the password for a user with the given email."""
    try:
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters long")
            
        # Enhanced password validation
        if not any(char.isupper() for char in new_password):
            raise ValueError("Password must contain at least one uppercase letter")
            
        if not any(char.islower() for char in new_password):
            raise ValueError("Password must contain at least one lowercase letter")
            
        if not any(char.isdigit() for char in new_password):
            raise ValueError("Password must contain at least one number")
            
        if not any(not char.isalnum() for char in new_password):
            raise ValueError("Password must contain at least one special character")
            
        password_hash = generate_password_hash(new_password)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?", 
            (password_hash, email.strip().lower())
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        raise Exception(f"Password reset failed: {str(e)}")

def get_user_by_email(email):
    """Retrieve a user by their email."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email = ?", 
            (email.strip().lower(),)
        )
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user by email: {str(e)}")
        return None

def get_user_by_id(user_id):
    """Retrieve a user by their ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE id = ?", 
            (user_id,)
        )
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user by ID: {str(e)}")
        return None

def get_user_by_name(name):
    """Retrieve a user by their name."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE name = ?", 
            (name.strip(),)
        )
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user by name: {str(e)}")
        return None

def update_user_profile(user_id, name, date_of_birth=None, gender=None, phone=None, address=None):
    """Update user profile information."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET name = ?, date_of_birth = ?, gender = ?, phone = ?, address = ?
            WHERE id = ?
        ''', (name.strip(), date_of_birth, gender, phone, address, user_id))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        raise Exception(f"Profile update failed: {str(e)}")

def delete_user_account(user_id):
    """Delete a user account by user ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete user history first (due to foreign key constraint)
        cursor.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
        
        # Then delete user
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        raise Exception(f"Account deletion failed: {str(e)}")

def delete_user_by_email(email):
    """Delete a user from the database by their email."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First get user_id to delete history
        cursor.execute("SELECT id FROM users WHERE email = ?", (email.strip().lower(),))
        user = cursor.fetchone()
        
        if user:
            user_id = user['id']
            # Delete user history
            cursor.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
            # Delete user
            cursor.execute("DELETE FROM users WHERE email = ?", (email.strip().lower(),))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return False

# ----------------------------
# User History Management (NEW)
# ----------------------------

def add_user_history(user_id, prediction_type, model_used=None, result=None, confidence=None, input_data=None):
    """Add a record to user history for tracking predictions and activities."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert input_data to JSON string if it's a dictionary
        if isinstance(input_data, dict):
            input_data = json.dumps(input_data)
        
        cursor.execute(
            '''INSERT INTO user_history 
               (user_id, prediction_type, model_used, result, confidence, input_data) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (user_id, prediction_type, model_used, result, confidence, input_data)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        print(f"Error adding user history: {str(e)}")
        return False

def get_user_history(user_id, limit=50):
    """Get user prediction history with pagination."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT * FROM user_history 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ?''',
            (user_id, limit)
        )
        history = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries and parse JSON input_data
        history_list = []
        for item in history:
            history_dict = dict(item)
            # Parse JSON input_data back to dict if it exists
            if history_dict.get('input_data'):
                try:
                    history_dict['input_data'] = json.loads(history_dict['input_data'])
                except json.JSONDecodeError:
                    # If parsing fails, keep as string
                    pass
            history_list.append(history_dict)
        
        return history_list
    except Exception as e:
        print(f"Error getting user history: {str(e)}")
        return []

def get_user_stats(user_id):
    """Get user statistics for profile page and analytics."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute(
            'SELECT COUNT(*) as count FROM user_history WHERE user_id = ?',
            (user_id,)
        )
        total_preds = cursor.fetchone()['count']
        
        # Predictions by type
        cursor.execute(
            '''SELECT prediction_type, COUNT(*) as count 
               FROM user_history 
               WHERE user_id = ? 
               GROUP BY prediction_type''',
            (user_id,)
        )
        type_stats = cursor.fetchall()
        
        # Most used model
        cursor.execute(
            '''SELECT model_used, COUNT(*) as count 
               FROM user_history 
               WHERE user_id = ? AND model_used IS NOT NULL
               GROUP BY model_used 
               ORDER BY count DESC 
               LIMIT 1''',
            (user_id,)
        )
        model_stats = cursor.fetchone()
        
        # Recent activity (last 5 entries)
        cursor.execute(
            '''SELECT prediction_type, created_at 
               FROM user_history 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT 5''',
            (user_id,)
        )
        recent_activity = cursor.fetchall()
        
        # Average confidence
        cursor.execute(
            '''SELECT AVG(confidence) as avg_confidence 
               FROM user_history 
               WHERE user_id = ? AND confidence IS NOT NULL''',
            (user_id,)
        )
        avg_confidence = cursor.fetchone()['avg_confidence']
        
        conn.close()
        
        return {
            'total_predictions': total_preds,
            'predictions_by_type': {row['prediction_type']: row['count'] for row in type_stats},
            'most_used_model': model_stats['model_used'] if model_stats else 'None',
            'recent_activity': [dict(row) for row in recent_activity],
            'average_confidence': round(avg_confidence * 100, 2) if avg_confidence else 0
        }
    except Exception as e:
        print(f"Error getting user stats: {str(e)}")
        return {}

def get_user_prediction_count(user_id, prediction_type=None):
    """Get count of predictions for a user, optionally filtered by type."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if prediction_type:
            cursor.execute(
                'SELECT COUNT(*) as count FROM user_history WHERE user_id = ? AND prediction_type = ?',
                (user_id, prediction_type)
            )
        else:
            cursor.execute(
                'SELECT COUNT(*) as count FROM user_history WHERE user_id = ?',
                (user_id,)
            )
        
        count = cursor.fetchone()['count']
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting prediction count: {str(e)}")
        return 0

def clear_user_history(user_id):
    """Clear all history for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_history WHERE user_id = ?', (user_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    except Exception as e:
        print(f"Error clearing user history: {str(e)}")
        return False

# ----------------------------
# Database Utilities
# ----------------------------

def get_all_users():
    """Get all users from the database (for admin purposes)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()
        return [dict(user) for user in users]
    except Exception as e:
        print(f"Error getting all users: {str(e)}")
        return []

def get_user_count():
    """Get the total number of users."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM users")
        count = cursor.fetchone()['count']
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting user count: {str(e)}")
        return 0

def backup_database():
    """Create a backup of the database."""
    try:
        if os.path.exists(DATABASE):
            backup_name = f"{DATABASE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(DATABASE, backup_name)
            return backup_name
        return None
    except Exception as e:
        print(f"Error creating database backup: {str(e)}")
        return None

# ----------------------------
# Additional Helper Functions
# ----------------------------

def validate_password_strength(password):
    """Validate password strength and return error message if weak."""
    if len(password) < 8:
        return "Password must be at least 8 characters long"
    
    if not any(char.isupper() for char in password):
        return "Password must contain at least one uppercase letter"
    
    if not any(char.islower() for char in password):
        return "Password must contain at least one lowercase letter"
    
    if not any(char.isdigit() for char in password):
        return "Password must contain at least one number"
    
    if not any(not char.isalnum() for char in password):
        return "Password must contain at least one special character"
    
    return None

def validate_email_format(email):
    """Validate email format."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

def get_user_statistics():
    """Get user statistics for admin dashboard."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total users
        cursor.execute("SELECT COUNT(*) as total FROM users")
        total_users = cursor.fetchone()['total']
        
        # Users by role
        cursor.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
        users_by_role = dict(cursor.fetchall())
        
        # New users this month
        cursor.execute("""
            SELECT COUNT(*) as count FROM users 
            WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
        """)
        new_this_month = cursor.fetchone()['count']
        
        # Total predictions across all users
        cursor.execute("SELECT COUNT(*) as total_predictions FROM user_history")
        total_predictions = cursor.fetchone()['total_predictions']
        
        # Predictions by type
        cursor.execute("SELECT prediction_type, COUNT(*) as count FROM user_history GROUP BY prediction_type")
        predictions_by_type = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_users': total_users,
            'users_by_role': users_by_role,
            'new_this_month': new_this_month,
            'total_predictions': total_predictions,
            'predictions_by_type': predictions_by_type
        }
    except Exception as e:
        print(f"Error getting user statistics: {str(e)}")
        return {}

def search_users(search_term):
    """Search users by name or email."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM users 
            WHERE name LIKE ? OR email LIKE ?
        """, (f'%{search_term}%', f'%{search_term}%'))
        
        users = cursor.fetchall()
        conn.close()
        return [dict(user) for user in users]
    except Exception as e:
        print(f"Error searching users: {str(e)}")
        return []

# ----------------------------
# Session Management Functions
# ----------------------------

def create_user_session_data(user):
    """Create session data for a logged-in user."""
    if not user:
        return None
    
    return {
        'user_id': user['id'],
        'user_name': user['name'],
        'user_email': user['email'],
        'user_role': user.get('role', 'user')
    }

def validate_session_user(user_id):
    """Validate if a user session is still valid."""
    try:
        user = get_user_by_id(user_id)
        return user is not None
    except Exception as e:
        print(f"Error validating session user: {str(e)}")
        return False

def update_user_session_name(user_id, new_name):
    """Update the user's name in active sessions (helper function)."""
    # This would typically be handled by the session management system
    # For Flask, we update the session directly in the route
    # This function serves as a placeholder for any additional logic needed
    try:
        user = get_user_by_id(user_id)
        if user:
            return True
        return False
    except Exception as e:
        print(f"Error updating session name: {str(e)}")
        return False

# ----------------------------
# Security Functions
# ----------------------------

def generate_secure_password(length=12):
    """Generate a secure random password."""
    import secrets
    import string
    
    if length < 8:
        length = 8
        
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special_chars = string.punctuation
    
    # Ensure the password has at least one of each type
    password = [
        secrets.choice(lowercase),
        secrets.choice(uppercase),
        secrets.choice(digits),
        secrets.choice(special_chars)
    ]
    
    # Fill the rest with random choices from all character sets
    all_chars = lowercase + uppercase + digits + special_chars
    password += [secrets.choice(all_chars) for _ in range(length - 4)]
    
    # Shuffle the password list
    secrets.SystemRandom().shuffle(password)
    
    return ''.join(password)

def verify_password_reset_token(token, email):
    """Verify a password reset token (simplified version)."""
    # In a real application, you would use a proper token system like itsdangerous
    # This is a simplified version for demonstration
    try:
        # Basic token validation (you should use a proper token library)
        if len(token) < 10:
            return False
            
        # Check if email exists
        user = get_user_by_email(email)
        if not user:
            return False
            
        # Here you would typically:
        # 1. Verify the token signature
        # 2. Check token expiration
        # 3. Validate token against stored hash
        
        # For now, we'll do a simple length check
        return len(token) >= 10
        
    except Exception as e:
        print(f"Error verifying password reset token: {str(e)}")
        return False

# ----------------------------
# User Profile Functions
# ----------------------------

def get_user_profile_completion(user_id):
    """Calculate user profile completion percentage."""
    try:
        user = get_user_by_id(user_id)
        if not user:
            return 0
            
        total_fields = 6  # name, email, date_of_birth, gender, phone, address
        completed_fields = 2  # name and email are always present
        
        if user.get('date_of_birth'):
            completed_fields += 1
        if user.get('gender'):
            completed_fields += 1
        if user.get('phone'):
            completed_fields += 1
        if user.get('address'):
            completed_fields += 1
            
        return int((completed_fields / total_fields) * 100)
        
    except Exception as e:
        print(f"Error calculating profile completion: {str(e)}")
        return 0

def get_recent_users(limit=10):
    """Get recently registered users."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM users 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        users = cursor.fetchall()
        conn.close()
        return [dict(user) for user in users]
    except Exception as e:
        print(f"Error getting recent users: {str(e)}")
        return []

def get_recent_activity(limit=20):
    """Get recent activity across all users (for admin dashboard)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT uh.*, u.name, u.email 
            FROM user_history uh
            JOIN users u ON uh.user_id = u.id
            ORDER BY uh.created_at DESC 
            LIMIT ?
        """, (limit,))
        activity = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries and parse JSON input_data
        activity_list = []
        for item in activity:
            activity_dict = dict(item)
            # Parse JSON input_data back to dict if it exists
            if activity_dict.get('input_data'):
                try:
                    activity_dict['input_data'] = json.loads(activity_dict['input_data'])
                except json.JSONDecodeError:
                    # If parsing fails, keep as string
                    pass
            activity_list.append(activity_dict)
        
        return activity_list
    except Exception as e:
        print(f"Error getting recent activity: {str(e)}")
        return []

# ----------------------------
# Analytics Functions (NEW)
# ----------------------------

def get_system_analytics():
    """Get comprehensive system analytics for admin dashboard."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # User statistics
        cursor.execute("SELECT COUNT(*) as total_users FROM users")
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute("SELECT COUNT(*) as active_today FROM users WHERE DATE(created_at) = DATE('now')")
        active_today = cursor.fetchone()['active_today']
        
        # Prediction statistics
        cursor.execute("SELECT COUNT(*) as total_predictions FROM user_history")
        total_predictions = cursor.fetchone()['total_predictions']
        
        cursor.execute("SELECT COUNT(*) as predictions_today FROM user_history WHERE DATE(created_at) = DATE('now')")
        predictions_today = cursor.fetchone()['predictions_today']
        
        # Prediction type distribution
        cursor.execute("SELECT prediction_type, COUNT(*) as count FROM user_history GROUP BY prediction_type")
        prediction_distribution = dict(cursor.fetchall())
        
        # Most active users
        cursor.execute("""
            SELECT u.name, u.email, COUNT(uh.id) as prediction_count 
            FROM users u 
            LEFT JOIN user_history uh ON u.id = uh.user_id 
            GROUP BY u.id 
            ORDER BY prediction_count DESC 
            LIMIT 10
        """)
        most_active_users = [dict(row) for row in cursor.fetchall()]
        
        # Daily predictions for last 7 days
        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count 
            FROM user_history 
            WHERE created_at >= DATE('now', '-7 days') 
            GROUP BY DATE(created_at) 
            ORDER BY date
        """)
        weekly_activity = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_users': total_users,
            'active_today': active_today,
            'total_predictions': total_predictions,
            'predictions_today': predictions_today,
            'prediction_distribution': prediction_distribution,
            'most_active_users': most_active_users,
            'weekly_activity': weekly_activity
        }
    except Exception as e:
        print(f"Error getting system analytics: {str(e)}")
        return {}

def export_user_data(user_id):
    """Export all user data including profile and history for GDPR compliance."""
    try:
        user = get_user_by_id(user_id)
        if not user:
            return None
            
        user_history = get_user_history(user_id, limit=1000)  # Get all history
        
        export_data = {
            'profile': user,
            'prediction_history': user_history,
            'exported_at': datetime.now().isoformat(),
            'total_predictions': len(user_history),
            'data_categories': ['profile', 'prediction_history']
        }
        
        return export_data
    except Exception as e:
        print(f"Error exporting user data: {str(e)}")
        return None

# ----------------------------
# Initialize Database on Import
# ----------------------------

# Initialize the database when this module is imported
init_database()
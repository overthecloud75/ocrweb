from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, PasswordField
from wtforms.fields.html5 import EmailField
from wtforms.validators import DataRequired, Length, EqualTo, Email

class SupervisingForm(FlaskForm):
    filename = StringField('filename', validators=[DataRequired(), Length(min=2, max=100)])
    order = StringField('order', validators=[DataRequired(), Length(min=1, max=100)])
    target = StringField('target', validators=[DataRequired(), Length(min=1, max=20)])
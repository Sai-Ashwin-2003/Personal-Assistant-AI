from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from .models import ChatSession

@receiver(user_logged_in)
def create_chat_session(sender, user, request, **kwargs):
    ChatSession.objects.create(user=user, title="New Chat")
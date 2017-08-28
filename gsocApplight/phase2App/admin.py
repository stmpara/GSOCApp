from django.contrib import admin

from .models import PostAd
from .models import TabAd


#class ItemAdmin(admin.ModelAdmin):
#list_display = ['category', 'location']

#admin.site.register(PostAd, ItemAdmin)
admin.site.register(PostAd)
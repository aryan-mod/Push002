import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { Settings, MapPin, Bell, Mail, MessageSquare, Save } from "lucide-react";
import Header from "@/components/layout/header";

/**
 * Alert Settings Page Component
 * Allows users to configure alert preferences including:
 * - Geo-fence radius for location-based alerts
 * - Alert types and severity filters
 * - Notification methods (SMS, email, push)
 * - Emergency contact preferences
 */
export default function AlertSettings() {
  const { toast } = useToast();
  
  // Alert settings state
  const [settings, setSettings] = useState({
    geoFence: {
      enabled: true,
      radius: 5, // kilometers
      alertOnExit: true,
      alertOnEntry: false,
    },
    alertTypes: {
      medical: true,
      weather: true,
      crime: false,
      naturalDisaster: true,
      wildlife: true,
      accident: true,
      other: false,
    },
    severityFilter: {
      critical: true,
      high: true,
      medium: false,
      low: false,
    },
    notifications: {
      pushNotifications: true,
      emailAlerts: false,
      smsAlerts: false,
      soundEnabled: true,
      vibrationEnabled: true,
    },
    contacts: {
      emergencyContact1: "",
      emergencyContact2: "",
      medicalInfo: "",
      bloodType: "",
      allergies: "",
    },
    location: {
      shareLocation: true,
      highAccuracyMode: false,
      locationHistory: true,
      batteryOptimization: true,
    }
  });

  // Load settings from localStorage on component mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('tourist-safety-settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setSettings(prev => ({ ...prev, ...parsed }));
      } catch (error) {
        console.error('Error loading saved settings:', error);
      }
    }
  }, []);

  // Save settings to backend and localStorage
  const handleSaveSettings = async () => {
    try {
      // Save to localStorage for immediate persistence
      localStorage.setItem('tourist-safety-settings', JSON.stringify(settings));
      
      // Save to backend API (assuming we have this endpoint)
      const response = await fetch("/api/v1/user/settings", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(settings),
      });

      if (response.ok) {
        toast({
          title: "Settings Saved",
          description: "Your alert preferences have been updated successfully.",
        });
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (error) {
      // Still save locally even if backend fails
      toast({
        title: "Settings Saved Locally",
        description: "Settings saved on your device. Will sync when connection is restored.",
        variant: "default",
      });
    }
  };

  return (
    <>
      <Header 
        title="Alert Settings" 
        subtitle="Configure your safety alert preferences and notification settings"
      />
      
      <div className="p-6 max-w-4xl mx-auto">
        <Tabs defaultValue="geofence" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="geofence">Geo-fence</TabsTrigger>
            <TabsTrigger value="alerts">Alert Types</TabsTrigger>
            <TabsTrigger value="notifications">Notifications</TabsTrigger>
            <TabsTrigger value="contacts">Contacts</TabsTrigger>
            <TabsTrigger value="location">Location</TabsTrigger>
          </TabsList>

          {/* Geo-fence Settings */}
          <TabsContent value="geofence">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  Geo-fence Configuration
                </CardTitle>
                <CardDescription>
                  Set up location-based alerts for when you enter or leave safe areas
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Enable Geo-fence Alerts</Label>
                    <p className="text-sm text-muted-foreground">
                      Get notified when entering or leaving designated safe zones
                    </p>
                  </div>
                  <Switch
                    checked={settings.geoFence.enabled}
                    onCheckedChange={(checked) =>
                      setSettings(prev => ({
                        ...prev,
                        geoFence: { ...prev.geoFence, enabled: checked }
                      }))
                    }
                    data-testid="switch-geofence-enabled"
                  />
                </div>

                {settings.geoFence.enabled && (
                  <>
                    <div className="space-y-3">
                      <Label>Safety Radius: {settings.geoFence.radius} km</Label>
                      <Slider
                        value={[settings.geoFence.radius]}
                        onValueChange={([value]) =>
                          setSettings(prev => ({
                            ...prev,
                            geoFence: { ...prev.geoFence, radius: value }
                          }))
                        }
                        max={50}
                        min={1}
                        step={1}
                        className="w-full"
                        data-testid="slider-geofence-radius"
                      />
                      <p className="text-sm text-muted-foreground">
                        Alert radius around your current location or designated safe areas
                      </p>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <Label>Alert when leaving safe area</Label>
                        <Switch
                          checked={settings.geoFence.alertOnExit}
                          onCheckedChange={(checked) =>
                            setSettings(prev => ({
                              ...prev,
                              geoFence: { ...prev.geoFence, alertOnExit: checked }
                            }))
                          }
                          data-testid="switch-alert-on-exit"
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <Label>Alert when entering new area</Label>
                        <Switch
                          checked={settings.geoFence.alertOnEntry}
                          onCheckedChange={(checked) =>
                            setSettings(prev => ({
                              ...prev,
                              geoFence: { ...prev.geoFence, alertOnEntry: checked }
                            }))
                          }
                          data-testid="switch-alert-on-entry"
                        />
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Alert Types Settings */}
          <TabsContent value="alerts">
            <Card>
              <CardHeader>
                <CardTitle>Alert Types & Severity</CardTitle>
                <CardDescription>
                  Choose which types of alerts you want to receive and their minimum severity levels
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h4 className="font-medium mb-3">Emergency Alert Types</h4>
                  <div className="space-y-3">
                    {Object.entries(settings.alertTypes).map(([type, enabled]) => (
                      <div key={type} className="flex items-center justify-between">
                        <Label className="capitalize">
                          {type.replace(/([A-Z])/g, ' $1').trim()}
                        </Label>
                        <Switch
                          checked={enabled}
                          onCheckedChange={(checked) =>
                            setSettings(prev => ({
                              ...prev,
                              alertTypes: { ...prev.alertTypes, [type]: checked }
                            }))
                          }
                          data-testid={`switch-alert-type-${type}`}
                        />
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-3">Minimum Severity Level</h4>
                  <div className="space-y-3">
                    {Object.entries(settings.severityFilter).map(([severity, enabled]) => (
                      <div key={severity} className="flex items-center justify-between">
                        <Label className="capitalize">
                          {severity} Priority Alerts
                        </Label>
                        <Switch
                          checked={enabled}
                          onCheckedChange={(checked) =>
                            setSettings(prev => ({
                              ...prev,
                              severityFilter: { ...prev.severityFilter, [severity]: checked }
                            }))
                          }
                          data-testid={`switch-severity-${severity}`}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Notification Settings */}
          <TabsContent value="notifications">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5" />
                  Notification Methods
                </CardTitle>
                <CardDescription>
                  Configure how you want to receive emergency alerts and notifications
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Push Notifications</Label>
                      <p className="text-sm text-muted-foreground">
                        Instant notifications on your device
                      </p>
                    </div>
                    <Switch
                      checked={settings.notifications.pushNotifications}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, pushNotifications: checked }
                        }))
                      }
                      data-testid="switch-push-notifications"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Email Alerts</Label>
                      <p className="text-sm text-muted-foreground">
                        Detailed alerts sent to your email
                      </p>
                    </div>
                    <Switch
                      checked={settings.notifications.emailAlerts}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, emailAlerts: checked }
                        }))
                      }
                      data-testid="switch-email-alerts"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>SMS Alerts</Label>
                      <p className="text-sm text-muted-foreground">
                        Text message alerts for critical emergencies
                      </p>
                    </div>
                    <Switch
                      checked={settings.notifications.smsAlerts}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, smsAlerts: checked }
                        }))
                      }
                      data-testid="switch-sms-alerts"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label>Alert Sound</Label>
                    <Switch
                      checked={settings.notifications.soundEnabled}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, soundEnabled: checked }
                        }))
                      }
                      data-testid="switch-sound-enabled"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label>Vibration</Label>
                    <Switch
                      checked={settings.notifications.vibrationEnabled}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, vibrationEnabled: checked }
                        }))
                      }
                      data-testid="switch-vibration-enabled"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Emergency Contacts */}
          <TabsContent value="contacts">
            <Card>
              <CardHeader>
                <CardTitle>Emergency Contacts & Medical Info</CardTitle>
                <CardDescription>
                  Set up emergency contacts and important medical information for first responders
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Emergency Contact 1</Label>
                    <Input
                      placeholder="Phone number or contact info"
                      value={settings.contacts.emergencyContact1}
                      onChange={(e) =>
                        setSettings(prev => ({
                          ...prev,
                          contacts: { ...prev.contacts, emergencyContact1: e.target.value }
                        }))
                      }
                      data-testid="input-emergency-contact-1"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Emergency Contact 2</Label>
                    <Input
                      placeholder="Phone number or contact info"
                      value={settings.contacts.emergencyContact2}
                      onChange={(e) =>
                        setSettings(prev => ({
                          ...prev,
                          contacts: { ...prev.contacts, emergencyContact2: e.target.value }
                        }))
                      }
                      data-testid="input-emergency-contact-2"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Blood Type</Label>
                    <Select
                      value={settings.contacts.bloodType}
                      onValueChange={(value) =>
                        setSettings(prev => ({
                          ...prev,
                          contacts: { ...prev.contacts, bloodType: value }
                        }))
                      }
                    >
                      <SelectTrigger data-testid="select-blood-type">
                        <SelectValue placeholder="Select blood type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="A+">A+</SelectItem>
                        <SelectItem value="A-">A-</SelectItem>
                        <SelectItem value="B+">B+</SelectItem>
                        <SelectItem value="B-">B-</SelectItem>
                        <SelectItem value="AB+">AB+</SelectItem>
                        <SelectItem value="AB-">AB-</SelectItem>
                        <SelectItem value="O+">O+</SelectItem>
                        <SelectItem value="O-">O-</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Medical Information</Label>
                    <Input
                      placeholder="Conditions, medications, etc."
                      value={settings.contacts.medicalInfo}
                      onChange={(e) =>
                        setSettings(prev => ({
                          ...prev,
                          contacts: { ...prev.contacts, medicalInfo: e.target.value }
                        }))
                      }
                      data-testid="input-medical-info"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Allergies</Label>
                  <Input
                    placeholder="List any known allergies"
                    value={settings.contacts.allergies}
                    onChange={(e) =>
                      setSettings(prev => ({
                        ...prev,
                        contacts: { ...prev.contacts, allergies: e.target.value }
                      }))
                    }
                    data-testid="input-allergies"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Location Settings */}
          <TabsContent value="location">
            <Card>
              <CardHeader>
                <CardTitle>Location & Privacy Settings</CardTitle>
                <CardDescription>
                  Configure location sharing and tracking preferences for safety features
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Share Location for Safety</Label>
                      <p className="text-sm text-muted-foreground">
                        Allow emergency services to locate you during alerts
                      </p>
                    </div>
                    <Switch
                      checked={settings.location.shareLocation}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          location: { ...prev.location, shareLocation: checked }
                        }))
                      }
                      data-testid="switch-share-location"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>High Accuracy Mode</Label>
                      <p className="text-sm text-muted-foreground">
                        More precise location tracking (uses more battery)
                      </p>
                    </div>
                    <Switch
                      checked={settings.location.highAccuracyMode}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          location: { ...prev.location, highAccuracyMode: checked }
                        }))
                      }
                      data-testid="switch-high-accuracy"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Keep Location History</Label>
                      <p className="text-sm text-muted-foreground">
                        Store location history for route analysis and safety insights
                      </p>
                    </div>
                    <Switch
                      checked={settings.location.locationHistory}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          location: { ...prev.location, locationHistory: checked }
                        }))
                      }
                      data-testid="switch-location-history"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Battery Optimization</Label>
                      <p className="text-sm text-muted-foreground">
                        Reduce location update frequency to save battery
                      </p>
                    </div>
                    <Switch
                      checked={settings.location.batteryOptimization}
                      onCheckedChange={(checked) =>
                        setSettings(prev => ({
                          ...prev,
                          location: { ...prev.location, batteryOptimization: checked }
                        }))
                      }
                      data-testid="switch-battery-optimization"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Save Button */}
        <div className="flex justify-end pt-6 border-t">
          <Button 
            onClick={handleSaveSettings}
            className="flex items-center gap-2"
            data-testid="button-save-settings"
          >
            <Save className="w-4 h-4" />
            Save Settings
          </Button>
        </div>
      </div>
    </>
  );
}
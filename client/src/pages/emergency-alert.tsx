import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { AlertTriangle, MapPin, Phone, Send } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import Header from "@/components/layout/header";

/**
 * Emergency Alert Page Component
 * Allows users to send emergency alerts with location and contact information
 * Integrates with backend API for alert creation and notification dispatch
 */
export default function EmergencyAlert() {
  const { toast } = useToast();
  const [alertData, setAlertData] = useState({
    type: "",
    description: "",
    location: "",
    contact: "",
    severity: "high"
  });

  // Get user's current location
  const [userLocation, setUserLocation] = useState<{lat: number, lng: number} | null>(null);

  // Fetch user's current location on component mount
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
          setAlertData(prev => ({
            ...prev,
            location: `${position.coords.latitude.toFixed(6)}, ${position.coords.longitude.toFixed(6)}`
          }));
        },
        (error) => {
          console.error("Error getting location:", error);
          toast({
            title: "Location Error",
            description: "Could not get your current location. Please enter manually.",
            variant: "destructive",
          });
        }
      );
    }
  }, [toast]);

  // Mutation for creating emergency alert
  const createAlertMutation = useMutation({
    mutationFn: async (alertData: any) => {
      return apiRequest("/api/v1/alerts", {
        method: "POST",
        body: alertData,
      });
    },
    onSuccess: () => {
      toast({
        title: "Emergency Alert Sent!",
        description: "Your emergency alert has been dispatched to relevant authorities.",
      });
      // Reset form
      setAlertData({
        type: "",
        description: "",
        location: userLocation ? `${userLocation.lat.toFixed(6)}, ${userLocation.lng.toFixed(6)}` : "",
        contact: "",
        severity: "high"
      });
      // Invalidate alerts cache to refresh notifications
      queryClient.invalidateQueries({ queryKey: ["/api/v1/alerts"] });
    },
    onError: (error) => {
      toast({
        title: "Failed to Send Alert",
        description: "There was an error sending your emergency alert. Please try again.",
        variant: "destructive",
      });
    },
  });

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!alertData.type || !alertData.description) {
      toast({
        title: "Missing Information",
        description: "Please fill in alert type and description.",
        variant: "destructive",
      });
      return;
    }

    // Create alert with tourist safety context
    createAlertMutation.mutate({
      siteId: "emergency", // Special site ID for tourist emergencies
      type: "emergency",
      severity: alertData.severity,
      title: `Tourist Emergency: ${alertData.type}`,
      message: `${alertData.description}${alertData.location ? ` | Location: ${alertData.location}` : ''}${alertData.contact ? ` | Contact: ${alertData.contact}` : ''}`,
      actionPlan: getTouristEmergencyActionPlan(alertData.type, alertData.severity),
      status: "active"
    });
  };

  return (
    <>
      <Header 
        title="Emergency Alert" 
        subtitle="Report emergencies and get immediate assistance"
      />
      
      <div className="p-6 max-w-2xl mx-auto">
        <Card className="border-red-200 bg-red-50 dark:bg-red-950/20">
          <CardHeader className="text-center">
            <div className="mx-auto w-16 h-16 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center mb-4">
              <AlertTriangle className="w-8 h-8 text-red-600" />
            </div>
            <CardTitle className="text-red-700 dark:text-red-300">Emergency Alert System</CardTitle>
            <CardDescription>
              Send immediate alerts for emergencies, medical situations, or safety concerns
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Alert Type Selection */}
              <div className="space-y-2">
                <Label htmlFor="alert-type">Emergency Type *</Label>
                <Select 
                  value={alertData.type} 
                  onValueChange={(value) => setAlertData(prev => ({...prev, type: value}))}
                >
                  <SelectTrigger data-testid="select-emergency-type">
                    <SelectValue placeholder="Select emergency type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="medical">Medical Emergency</SelectItem>
                    <SelectItem value="accident">Accident</SelectItem>
                    <SelectItem value="lost">Lost/Missing Person</SelectItem>
                    <SelectItem value="natural_disaster">Natural Disaster</SelectItem>
                    <SelectItem value="crime">Crime/Security</SelectItem>
                    <SelectItem value="weather">Severe Weather</SelectItem>
                    <SelectItem value="wildlife">Wildlife Encounter</SelectItem>
                    <SelectItem value="other">Other Emergency</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Severity Level */}
              <div className="space-y-2">
                <Label htmlFor="severity">Severity Level</Label>
                <Select 
                  value={alertData.severity} 
                  onValueChange={(value) => setAlertData(prev => ({...prev, severity: value}))}
                >
                  <SelectTrigger data-testid="select-severity">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="critical">Critical - Life Threatening</SelectItem>
                    <SelectItem value="high">High - Immediate Assistance Needed</SelectItem>
                    <SelectItem value="medium">Medium - Assistance Required</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Description */}
              <div className="space-y-2">
                <Label htmlFor="description">Description *</Label>
                <Textarea
                  id="description"
                  placeholder="Describe the emergency situation in detail..."
                  value={alertData.description}
                  onChange={(e) => setAlertData(prev => ({...prev, description: e.target.value}))}
                  className="min-h-24"
                  data-testid="input-description"
                />
              </div>

              {/* Location */}
              <div className="space-y-2">
                <Label htmlFor="location" className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  Current Location
                </Label>
                <Input
                  id="location"
                  placeholder="Auto-detected location or enter manually"
                  value={alertData.location}
                  onChange={(e) => setAlertData(prev => ({...prev, location: e.target.value}))}
                  data-testid="input-location"
                />
                {userLocation && (
                  <p className="text-sm text-muted-foreground">
                    📍 Location auto-detected from GPS
                  </p>
                )}
              </div>

              {/* Contact Information */}
              <div className="space-y-2">
                <Label htmlFor="contact" className="flex items-center gap-2">
                  <Phone className="w-4 h-4" />
                  Contact Information
                </Label>
                <Input
                  id="contact"
                  placeholder="Phone number or alternative contact method"
                  value={alertData.contact}
                  onChange={(e) => setAlertData(prev => ({...prev, contact: e.target.value}))}
                  data-testid="input-contact"
                />
              </div>

              {/* Submit Button */}
              <Button 
                type="submit" 
                className="w-full bg-red-600 hover:bg-red-700 text-white"
                disabled={createAlertMutation.isPending}
                data-testid="button-send-alert"
              >
                {createAlertMutation.isPending ? (
                  <>Sending Alert...</>
                ) : (
                  <>
                    <Send className="w-4 h-4 mr-2" />
                    Send Emergency Alert
                  </>
                )}
              </Button>
            </form>

            {/* Emergency Instructions */}
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
              <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">
                Emergency Instructions:
              </h4>
              <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                <li>• For life-threatening emergencies, call local emergency services first</li>
                <li>• This alert will notify relevant authorities and nearby safety coordinators</li>
                <li>• Stay calm and follow any instructions received</li>
                <li>• Keep your phone available for follow-up contact</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}

/**
 * Generate appropriate action plan based on emergency type and severity
 */
function getTouristEmergencyActionPlan(type: string, severity: string): string {
  const plans: Record<string, Record<string, string>> = {
    medical: {
      critical: "1. Call local emergency services immediately\n2. Dispatch nearest medical team\n3. Coordinate with hospitals\n4. Notify embassy if foreign tourist",
      high: "1. Send medical assistance\n2. Contact nearest clinic\n3. Provide first aid guidance\n4. Monitor situation",
      medium: "1. Connect with local medical support\n2. Provide medical advice\n3. Arrange transportation if needed"
    },
    lost: {
      critical: "1. Launch immediate search operation\n2. Alert all rescue teams\n3. Coordinate with local authorities\n4. Set up communication center",
      high: "1. Initiate search protocols\n2. Contact local guides\n3. Alert nearby tourists\n4. Monitor last known location",
      medium: "1. Verify location details\n2. Provide navigation assistance\n3. Contact local support"
    },
    accident: {
      critical: "1. Send emergency response team\n2. Clear evacuation route\n3. Alert medical services\n4. Secure accident site",
      high: "1. Dispatch first responders\n2. Provide immediate assistance\n3. Document incident\n4. Contact insurance",
      medium: "1. Assess situation remotely\n2. Provide guidance\n3. Arrange local assistance"
    }
  };

  return plans[type]?.[severity] || "1. Assess situation\n2. Provide appropriate assistance\n3. Monitor developments\n4. Follow up with tourist";
}
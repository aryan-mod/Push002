import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Header from "@/components/layout/header";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Upload, Image, Database, Satellite } from "lucide-react";

export default function DataUpload() {
  const { toast } = useToast();
  const [uploadProgress, setUploadProgress] = useState<{ [key: string]: number }>({});

  const uploadMutation = useMutation({
    mutationFn: async (data: any) => {
      return apiRequest("POST", "/api/v1/ingest", data);
    },
    onSuccess: () => {
      toast({
        title: "Upload Successful",
        description: "Data has been successfully uploaded and processed.",
      });
    },
    onError: () => {
      toast({
        title: "Upload Failed",
        description: "Failed to upload data. Please check the format and try again.",
        variant: "destructive",
      });
    },
  });

  const handleFileUpload = (file: File, type: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    // Simulate upload progress
    const uploadId = Math.random().toString(36).substr(2, 9);
    setUploadProgress(prev => ({ ...prev, [uploadId]: 0 }));

    const interval = setInterval(() => {
      setUploadProgress(prev => {
        const current = prev[uploadId] || 0;
        if (current >= 100) {
          clearInterval(interval);
          return prev;
        }
        return { ...prev, [uploadId]: current + 10 };
      });
    }, 200);

    // Simulate completion after 2 seconds
    setTimeout(() => {
      setUploadProgress(prev => ({ ...prev, [uploadId]: 100 }));
      clearInterval(interval);
    }, 2000);
  };

  return (
    <div className="flex flex-col">
      <Header title="Data Upload" subtitle="Upload sensor data, drone images, and DEM files" />
      
      <div className="flex-1 p-6">
        <Tabs defaultValue="sensor" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="sensor" data-testid="tab-sensor">
              <Database className="h-4 w-4 mr-2" />
              Sensor Data
            </TabsTrigger>
            <TabsTrigger value="drone" data-testid="tab-drone">
              <Image className="h-4 w-4 mr-2" />
              Drone Images
            </TabsTrigger>
            <TabsTrigger value="dem" data-testid="tab-dem">
              <Satellite className="h-4 w-4 mr-2" />
              DEM Files
            </TabsTrigger>
            <TabsTrigger value="batch" data-testid="tab-batch">
              <Upload className="h-4 w-4 mr-2" />
              Batch Upload
            </TabsTrigger>
          </TabsList>

          <TabsContent value="sensor" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Manual Entry */}
              <Card>
                <CardHeader>
                  <CardTitle>Manual Sensor Reading</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="sensor-id">Sensor ID</Label>
                      <Input id="sensor-id" placeholder="Enter sensor ID" data-testid="input-sensor-id" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="sensor-type">Sensor Type</Label>
                      <Select>
                        <SelectTrigger data-testid="select-sensor-type">
                          <SelectValue placeholder="Select type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="strain">Strain</SelectItem>
                          <SelectItem value="displacement">Displacement</SelectItem>
                          <SelectItem value="pore_pressure">Pore Pressure</SelectItem>
                          <SelectItem value="tilt">Tilt</SelectItem>
                          <SelectItem value="vibration">Vibration</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="value">Value</Label>
                      <Input id="value" type="number" placeholder="Reading value" data-testid="input-value" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="unit">Unit</Label>
                      <Input id="unit" placeholder="e.g., µε, mm, kPa" data-testid="input-unit" />
                    </div>
                  </div>
                  <Button 
                    className="w-full" 
                    disabled={uploadMutation.isPending}
                    data-testid="button-submit-reading"
                  >
                    Submit Reading
                  </Button>
                </CardContent>
              </Card>

              {/* CSV Upload */}
              <Card>
                <CardHeader>
                  <CardTitle>CSV File Upload</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                    <Database className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Drop CSV file here or click to upload
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Format: sensor_id, timestamp, value, unit, quality
                    </p>
                  </div>
                  <Button variant="outline" className="w-full" data-testid="button-upload-csv">
                    Choose CSV File
                  </Button>
                  
                  {/* Progress indicators */}
                  {Object.entries(uploadProgress).map(([id, progress]) => (
                    <div key={id} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Uploading...</span>
                        <span>{progress}%</span>
                      </div>
                      <Progress value={progress} />
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="drone" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Image Upload */}
              <Card>
                <CardHeader>
                  <CardTitle>Drone Image Upload</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="site-select">Site</Label>
                    <Select>
                      <SelectTrigger data-testid="select-site">
                        <SelectValue placeholder="Select site" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="site-a127">Site A-127</SelectItem>
                        <SelectItem value="site-b089">Site B-089</SelectItem>
                        <SelectItem value="site-c045">Site C-045</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                    <Image className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Drop images here or click to upload
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supported: JPG, PNG, TIFF (max 50MB each)
                    </p>
                  </div>
                  
                  <Button variant="outline" className="w-full" data-testid="button-upload-images">
                    Choose Images
                  </Button>
                </CardContent>
              </Card>

              {/* Metadata */}
              <Card>
                <CardHeader>
                  <CardTitle>Image Metadata</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="altitude">Altitude (m)</Label>
                      <Input id="altitude" type="number" placeholder="Flight altitude" data-testid="input-altitude" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="weather">Weather</Label>
                      <Select>
                        <SelectTrigger data-testid="select-weather">
                          <SelectValue placeholder="Select weather" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="clear">Clear</SelectItem>
                          <SelectItem value="cloudy">Cloudy</SelectItem>
                          <SelectItem value="rainy">Rainy</SelectItem>
                          <SelectItem value="foggy">Foggy</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="notes">Notes</Label>
                    <Textarea 
                      id="notes" 
                      placeholder="Additional observations..." 
                      data-testid="textarea-notes"
                    />
                  </div>
                  
                  <Button className="w-full" data-testid="button-save-metadata">
                    Save Metadata
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="dem" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* DEM Upload */}
              <Card>
                <CardHeader>
                  <CardTitle>DEM File Upload</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                    <Satellite className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Drop DEM files here or click to upload
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supported: GeoTIFF, ASCII Grid (max 500MB)
                    </p>
                  </div>
                  <Button variant="outline" className="w-full" data-testid="button-upload-dem">
                    Choose DEM Files
                  </Button>
                </CardContent>
              </Card>

              {/* Processing Options */}
              <Card>
                <CardHeader>
                  <CardTitle>Processing Options</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Auto-process features</Label>
                    <div className="space-y-2">
                      <label className="flex items-center space-x-2">
                        <input type="checkbox" className="rounded" defaultChecked />
                        <span className="text-sm">Calculate slope</span>
                      </label>
                      <label className="flex items-center space-x-2">
                        <input type="checkbox" className="rounded" defaultChecked />
                        <span className="text-sm">Calculate aspect</span>
                      </label>
                      <label className="flex items-center space-x-2">
                        <input type="checkbox" className="rounded" />
                        <span className="text-sm">Generate contours</span>
                      </label>
                      <label className="flex items-center space-x-2">
                        <input type="checkbox" className="rounded" />
                        <span className="text-sm">Calculate hillshade</span>
                      </label>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="resolution">Output Resolution (m)</Label>
                    <Input id="resolution" type="number" defaultValue="1" data-testid="input-resolution" />
                  </div>
                  
                  <Button className="w-full" data-testid="button-start-processing">
                    Start Processing
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="batch" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Batch Data Upload</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Sensor Data Batch */}
                  <div className="space-y-4">
                    <h3 className="font-medium">Sensor Data</h3>
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4 text-center">
                      <Database className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                      <p className="text-xs text-muted-foreground">CSV files</p>
                    </div>
                    <Button variant="outline" size="sm" className="w-full" data-testid="button-batch-sensor">
                      Upload Sensor CSVs
                    </Button>
                  </div>

                  {/* Images Batch */}
                  <div className="space-y-4">
                    <h3 className="font-medium">Drone Images</h3>
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4 text-center">
                      <Image className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                      <p className="text-xs text-muted-foreground">Image files</p>
                    </div>
                    <Button variant="outline" size="sm" className="w-full" data-testid="button-batch-images">
                      Upload Images
                    </Button>
                  </div>

                  {/* DEM Batch */}
                  <div className="space-y-4">
                    <h3 className="font-medium">DEM Files</h3>
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4 text-center">
                      <Satellite className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                      <p className="text-xs text-muted-foreground">GeoTIFF files</p>
                    </div>
                    <Button variant="outline" size="sm" className="w-full" data-testid="button-batch-dem">
                      Upload DEM Files
                    </Button>
                  </div>
                </div>

                {/* Upload Queue */}
                <div className="space-y-4">
                  <h3 className="font-medium">Upload Queue</h3>
                  <div className="border rounded-lg p-4">
                    <p className="text-sm text-muted-foreground text-center">
                      No files in queue
                    </p>
                  </div>
                </div>

                <Button className="w-full" data-testid="button-process-batch">
                  Process All Files
                </Button>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

import { storage } from "../storage";
import type { Alert } from "@shared/schema";

interface NotificationChannel {
  type: "sms" | "email" | "whatsapp" | "push";
  recipients: string[];
}

class AlertService {
  private twilioAccountSid = process.env.TWILIO_ACCOUNT_SID;
  private twilioAuthToken = process.env.TWILIO_AUTH_TOKEN;
  private twilioPhoneNumber = process.env.TWILIO_PHONE_NUMBER;
  private smtpConfig = {
    host: process.env.SMTP_HOST,
    port: parseInt(process.env.SMTP_PORT || "587"),
    user: process.env.SMTP_USER,
    password: process.env.SMTP_PASSWORD
  };

  async sendAlert(alert: Alert): Promise<void> {
    try {
      // Get notification channels based on severity
      const channels = this.getNotificationChannels(alert.severity);
      
      for (const channel of channels) {
        await this.sendNotificationBatch(alert, channel);
      }
    } catch (error) {
      console.error("Alert sending failed:", error);
    }
  }

  private getNotificationChannels(severity: string): NotificationChannel[] {
    const channels: NotificationChannel[] = [];

    switch (severity) {
      case "critical":
        channels.push(
          { type: "sms", recipients: this.getCriticalContactsSMS() },
          { type: "email", recipients: this.getCriticalContactsEmail() },
          { type: "whatsapp", recipients: this.getCriticalContactsWhatsApp() },
          { type: "push", recipients: this.getAllUsers() }
        );
        break;
      
      case "high":
        channels.push(
          { type: "sms", recipients: this.getHighPriorityContactsSMS() },
          { type: "email", recipients: this.getAllContactsEmail() },
          { type: "push", recipients: this.getAllUsers() }
        );
        break;
      
      case "medium":
        channels.push(
          { type: "email", recipients: this.getAllContactsEmail() },
          { type: "push", recipients: this.getAllUsers() }
        );
        break;
      
      default: // low
        channels.push(
          { type: "push", recipients: this.getAllUsers() }
        );
    }

    return channels;
  }

  private async sendNotificationBatch(alert: Alert, channel: NotificationChannel): Promise<void> {
    for (const recipient of channel.recipients) {
      try {
        const notification = await storage.createAlertNotification({
          alertId: alert.id,
          channel: channel.type,
          recipient,
          status: "pending"
        });

        switch (channel.type) {
          case "sms":
            await this.sendSMS(recipient, alert);
            break;
          case "email":
            await this.sendEmail(recipient, alert);
            break;
          case "whatsapp":
            await this.sendWhatsApp(recipient, alert);
            break;
          case "push":
            await this.sendPushNotification(recipient, alert);
            break;
        }

        await storage.updateNotificationStatus(notification.id, "sent");
      } catch (error) {
        console.error(`Failed to send ${channel.type} to ${recipient}:`, error);
        // Update status to failed if we have the notification ID
      }
    }
  }

  private async sendSMS(phoneNumber: string, alert: Alert): Promise<void> {
    if (!this.twilioAccountSid || !this.twilioAuthToken) {
      console.warn("Twilio credentials not configured");
      return;
    }

    const message = this.formatSMSMessage(alert);
    
    // Simulate Twilio SMS API call
    const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${this.twilioAccountSid}/Messages.json`, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${Buffer.from(`${this.twilioAccountSid}:${this.twilioAuthToken}`).toString('base64')}`,
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: new URLSearchParams({
        From: this.twilioPhoneNumber || "",
        To: phoneNumber,
        Body: message
      })
    });

    if (!response.ok) {
      throw new Error(`SMS sending failed: ${response.statusText}`);
    }
  }

  private async sendEmail(email: string, alert: Alert): Promise<void> {
    const subject = `🚨 Rockfall Alert: ${alert.title}`;
    const body = this.formatEmailMessage(alert);

    // Simulate SMTP email sending
    console.log(`Sending email to ${email}: ${subject}`);
    
    // In a real implementation, you would use nodemailer or similar
    // const transporter = nodemailer.createTransporter(this.smtpConfig);
    // await transporter.sendMail({ to: email, subject, html: body });
  }

  private async sendWhatsApp(phoneNumber: string, alert: Alert): Promise<void> {
    if (!this.twilioAccountSid || !this.twilioAuthToken) {
      console.warn("Twilio credentials not configured");
      return;
    }

    const message = this.formatWhatsAppMessage(alert);
    
    // Simulate Twilio WhatsApp API call
    const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${this.twilioAccountSid}/Messages.json`, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${Buffer.from(`${this.twilioAccountSid}:${this.twilioAuthToken}`).toString('base64')}`,
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: new URLSearchParams({
        From: `whatsapp:${this.twilioPhoneNumber}`,
        To: `whatsapp:${phoneNumber}`,
        Body: message
      })
    });

    if (!response.ok) {
      throw new Error(`WhatsApp sending failed: ${response.statusText}`);
    }
  }

  private async sendPushNotification(userId: string, alert: Alert): Promise<void> {
    // Simulate web push notification
    const payload = {
      title: alert.title,
      body: alert.message,
      icon: '/icon-192x192.png',
      badge: '/badge-72x72.png',
      tag: `alert-${alert.id}`,
      data: {
        alertId: alert.id,
        siteId: alert.siteId,
        severity: alert.severity
      }
    };

    console.log(`Sending push notification to user ${userId}:`, payload);
    
    // In a real implementation, you would use web-push library
    // webpush.sendNotification(subscription, JSON.stringify(payload));
  }

  private formatSMSMessage(alert: Alert): string {
    return `🚨 ROCKFALL ALERT\n${alert.title}\n${alert.message}\nAction: ${alert.actionPlan?.split('\n')[0] || 'Monitor situation'}`;
  }

  private formatEmailMessage(alert: Alert): string {
    return `
      <h2>🚨 Rockfall Prediction Alert</h2>
      <h3>${alert.title}</h3>
      <p><strong>Severity:</strong> ${alert.severity.toUpperCase()}</p>
      <p><strong>Message:</strong> ${alert.message}</p>
      ${alert.actionPlan ? `
        <h4>Recommended Actions:</h4>
        <pre>${alert.actionPlan}</pre>
      ` : ''}
      <p><small>Alert generated at: ${alert.createdAt}</small></p>
      <p><em>RockWatch AI Prediction System</em></p>
    `;
  }

  private formatWhatsAppMessage(alert: Alert): string {
    return `🚨 *ROCKFALL ALERT*\n\n*${alert.title}*\n\n${alert.message}\n\n*Action Required:*\n${alert.actionPlan?.split('\n')[0] || 'Monitor situation'}\n\n_RockWatch AI System_`;
  }

  // Contact management methods (in real implementation, these would be configurable)
  private getCriticalContactsSMS(): string[] {
    return [
      process.env.EMERGENCY_CONTACT_1 || "+1234567890",
      process.env.EMERGENCY_CONTACT_2 || "+1234567891"
    ];
  }

  private getCriticalContactsEmail(): string[] {
    return [
      process.env.EMERGENCY_EMAIL_1 || "emergency@example.com",
      process.env.EMERGENCY_EMAIL_2 || "supervisor@example.com"
    ];
  }

  private getCriticalContactsWhatsApp(): string[] {
    return [
      process.env.EMERGENCY_WHATSAPP_1 || "+1234567890",
      process.env.EMERGENCY_WHATSAPP_2 || "+1234567891"
    ];
  }

  private getHighPriorityContactsSMS(): string[] {
    return [
      ...this.getCriticalContactsSMS(),
      process.env.SUPERVISOR_CONTACT || "+1234567892"
    ];
  }

  private getAllContactsEmail(): string[] {
    return [
      ...this.getCriticalContactsEmail(),
      process.env.TEAM_EMAIL || "team@example.com",
      process.env.OPERATIONS_EMAIL || "operations@example.com"
    ];
  }

  private getAllUsers(): string[] {
    // In real implementation, fetch from database
    return ["user1", "user2", "user3"];
  }
}

export const alertService = new AlertService();

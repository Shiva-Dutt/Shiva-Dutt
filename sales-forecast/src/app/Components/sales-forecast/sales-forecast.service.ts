import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MatSnackBar } from '@angular/material/snack-bar';

@Injectable({
  providedIn: 'root'
})
export class SalesForecastService {

  baseApiUrl  = "http://localhost:5000/forecast"

  constructor(private snackBar: MatSnackBar, private http: HttpClient) { }

  upload(data: any){
    this.snackBar.open("Forecast period is selected", "Ok", { duration: 5000 });
    return this.http.post(
      this.baseApiUrl,
      JSON.stringify(data)
    )
  }
}

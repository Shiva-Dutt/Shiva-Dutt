import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { MatSnackBar } from '@angular/material/snack-bar';

@Injectable({
  providedIn: 'root'
})
export class FileUploadService {

  baseApiUrl  = "http://localhost:5000/upload"

  constructor(private snackBar: MatSnackBar, private http: HttpClient) { }

  upload(file: any) :Observable<any>{

      const formData = new FormData();

      formData.append("file", file, file.name);
      this.snackBar.open("File successfully uploaded", "Ok", { duration: 5000 });
      return this.http.post(this.baseApiUrl, formData)
  }


}
